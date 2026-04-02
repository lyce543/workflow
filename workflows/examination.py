from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace
from openai.types.responses import ResponseTextDeltaEvent

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class ExaminationWorkflow(BaseWorkflow):

    def create_interviewer_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context

            if ctx.state.current_question_index >= len(ctx.state.questions):
                return "The exam is complete. Thank the student."

            current_q = ctx.state.questions[ctx.state.current_question_index]
            pending = ctx.state.custom_data.get('pending') or {}
            is_followup = pending.get('is_follow_up', False)
            last_user_answer = ctx.state.custom_data.get('last_user_answer', '')

            if is_followup:
                return f"""You are conducting an oral exam. The student gave a partial answer.

Question: {current_q['question']}
Student's previous answer: {last_user_answer}

Ask a NATURAL follow-up question that:
- Encourages the student to elaborate or clarify
- Does NOT reveal the correct answer or key concepts
- Uses open-ended phrasing like:
  * "Чи можете розповісти більше про..."
  * "Що ще ви знаєте про..."
  * "Уточніть, будь ласка..."

Speak in Ukrainian. Be supportive but neutral."""
            else:
                return f"""You are an examiner conducting an oral exam.

Current question: {current_q['question']}

Ask this question clearly and directly in Ukrainian.
Do NOT give hints or reveal key concepts.
Be professional and neutral."""

        return Agent[WorkflowContext](
            name="Interviewer",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, top_p=1, max_tokens=512)
        )

    def create_evaluator_agent(self, context: WorkflowContext, user_answer: str, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            current_q = ctx.state.questions[ctx.state.current_question_index]

            return f"""You are an evaluator for an oral examination.

QUESTION: {current_q['question']}
KEY CONCEPTS: {current_q['key_concepts']}
STUDENT ANSWER: {user_answer}

EVALUATION RULES:
1. Check if the answer SEMANTICALLY covers the key concepts
2. Accept synonyms, paraphrases, and detailed explanations
3. Focus on MEANING, not exact wording
4. If answer is clearly wrong or irrelevant → complete=false, needs_clarification=false
5. If answer partially addresses the topic → complete=false, needs_clarification=true
6. If answer fully covers the key concept (even with different words) → complete=true

Return JSON:
{{
  "complete": true/false,
  "missing_concepts": ["concept1", ...],
  "needs_clarification": true/false
}}

Current follow-up count: {ctx.state.follow_up_count}/{ctx.state.max_follow_ups}
If follow_up_count >= max, set needs_clarification=false even if incomplete."""

        class EvalOutput(BaseModel):
            complete: bool
            missing_concepts: List[str]
            needs_clarification: bool

        return Agent[WorkflowContext](
            name="Evaluator",
            instructions=agent_instructions,
            model=model,
            output_type=EvalOutput,
            model_settings=ModelSettings(temperature=0.2, max_tokens=512)
        )

    async def _ask_question(self, state: WorkflowState, context: WorkflowContext, model: str, is_follow_up: bool = False) -> AsyncGenerator[str, None]:
        interviewer = self.create_interviewer_agent(context, model)
        prompt = "Student answer was incomplete. Ask a follow-up question to clarify." if is_follow_up else ""
        result = Runner.run_streamed(interviewer, prompt, context=context)

        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta
                full_response += chunk
                yield chunk

        state.custom_data['pending'] = {
            "question_index": state.current_question_index,
            "interviewer_question": full_response,
            "is_follow_up": is_follow_up
        }

    def _migrate_old_answers(self, state: WorkflowState):
        """Migrate old state.answers format to custom_data['pending'] + custom_data['evaluations']."""
        old_answers = state.answers
        if not old_answers:
            return

        evaluations = {}
        pending = None

        for ans in old_answers:
            q_idx = ans.get('question_index', 0)
            answer_text = ans.get('answer', '')
            interviewer_question = ans.get('interviewer_question', '')
            evaluation = ans.get('evaluation', {})

            if answer_text:
                # Completed answer
                evaluations[str(q_idx)] = {
                    **evaluation,
                    "answer": answer_text,
                    "interviewer_question": interviewer_question
                }
            elif interviewer_question:
                # Question was asked but not answered yet
                pending = {
                    "question_index": q_idx,
                    "interviewer_question": interviewer_question,
                    "is_follow_up": ans.get('is_follow_up', False)
                }

        state.custom_data['evaluations'] = evaluations
        state.custom_data['pending'] = pending

    async def run_workflow_stream(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> AsyncGenerator[str, None]:
        with trace(f"Examination-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)

            if state.status == "finished":
                yield "Іспит вже завершено."
                return

            if state.current_question_index >= len(state.questions):
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                yield "Вітаю! Ви відповіли на всі питання. Іспит завершено."
                return

            # Migrate old format before overwriting state.answers with air history
            if 'pending' not in state.custom_data and 'evaluations' not in state.custom_data:
                self._migrate_old_answers(state)

            air_records = await xano.get_air_history(ub_id)
            state.answers = self._convert_air_to_history(air_records)

            context = WorkflowContext(state=state)
            model = template.get("model", "gpt-4o")

            # No pending question → ask the next question
            if not state.custom_data.get('pending'):
                async for chunk in self._ask_question(state, context, model):
                    yield chunk
                state.follow_up_count = 0
                await xano.save_workflow_state(state)
                return

            # Pending question exists → user is answering it
            state.custom_data['last_user_answer'] = user_message

            evaluator = self.create_evaluator_agent(context, user_message, model)
            eval_result = await Runner.run(evaluator, "", context=context)
            evaluation = eval_result.final_output.model_dump()

            # Store evaluation result keyed by question index
            evaluations = state.custom_data.get('evaluations', {})
            q_idx = str(state.current_question_index)
            evaluations[q_idx] = {
                **evaluation,
                "answer": user_message,
                "interviewer_question": state.custom_data['pending'].get('interviewer_question', '')
            }
            state.custom_data['evaluations'] = evaluations
            state.custom_data['pending'] = None

            if evaluation['complete']:
                state.current_question_index += 1
                state.follow_up_count = 0

                if state.current_question_index >= len(state.questions):
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    yield "Вітаю! Ви відповіли на всі питання. Іспит завершено."
                    return

                async for chunk in self._ask_question(state, context, model):
                    yield chunk
                await xano.save_workflow_state(state)

            else:
                if state.follow_up_count >= state.max_follow_ups:
                    state.current_question_index += 1
                    state.follow_up_count = 0

                    if state.current_question_index >= len(state.questions):
                        state.status = "finished"
                        await xano.save_workflow_state(state)
                        from models import ChatStatus
                        await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                        yield "Іспит завершено."
                        return

                    async for chunk in self._ask_question(state, context, model):
                        yield chunk
                    await xano.save_workflow_state(state)

                else:
                    state.follow_up_count += 1
                    async for chunk in self._ask_question(state, context, model, is_follow_up=True):
                        yield chunk
                    await xano.save_workflow_state(state)

    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"ExaminationEval-{ub_id}"):
            # Build answers list from stored evaluations in custom_data
            evaluations = workflow_state.custom_data.get('evaluations', {})
            if evaluations:
                answers = []
                for q_idx in sorted(evaluations.keys(), key=lambda x: int(x)):
                    ev = evaluations[q_idx]
                    answers.append({
                        "interviewer_question": ev.get("interviewer_question", ""),
                        "answer": ev.get("answer", ""),
                        "evaluation": {
                            "complete": ev.get("complete", False),
                            "missing_concepts": ev.get("missing_concepts", []),
                            "needs_clarification": ev.get("needs_clarification", False)
                        }
                    })
                workflow_state.answers = answers
            elif not workflow_state.answers:
                workflow_state.answers = []
            # else: keep workflow_state.answers populated from air (fallback for old migrated chats)

            context = EvaluationContext(
                workflow_state=workflow_state,
                eval_instructions=eval_instructions,
                criteria=criteria
            )

            total_max_points = self._calculate_total_points(criteria)

            def agent_instructions(run_context: RunContextWrapper[EvaluationContext], _agent: Agent):
                ctx = run_context.context

                criteria_text = ""
                for i, crit in enumerate(ctx.criteria):
                    criteria_text += f"\n## Criterion {i+1}"
                    if crit.get('name'):
                        criteria_text += f": {crit['name']}"
                    criteria_text += f"\n- Description: {crit.get('description', 'No description')}"
                    criteria_text += f"\n- Max points: {crit.get('max_points', 10)}"
                    criteria_text += "\n"

                answers_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    answers_text += f"\n### Question {i+1}"
                    if ans.get('interviewer_question'):
                        answers_text += f"\nAI Question: {ans['interviewer_question']}"
                    answers_text += f"\nStudent answer: {ans.get('answer', 'No answer')}"
                    if ans.get('evaluation'):
                        answers_text += f"\nEvaluation: complete={ans['evaluation'].get('complete', False)}"
                    answers_text += "\n"

                return f"""You are evaluating an oral examination.

# Custom Instructions
{ctx.eval_instructions}

# Evaluation Criteria (Total max: {total_max_points} points)
{criteria_text}

# Student's Answers
{answers_text}

# Your Task
1. Evaluate each answer against the criteria
2. Provide specific feedback for each criterion
3. Calculate total points (max {total_max_points})

Format your response as:
## Overall Assessment
[Brief summary]

## Criterion Scores
[For each criterion: score/max and justification]

## Total Score: X/{total_max_points}

## Recommendations
[Specific advice for improvement]"""

            agent = Agent[EvaluationContext](
                name="ExamEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )

            result = await Runner.run(agent, "", context=context)
            return result.final_output
