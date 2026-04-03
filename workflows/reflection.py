from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel
import json

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace
from openai.types.responses import ResponseTextDeltaEvent

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class ReflectionWorkflow(BaseWorkflow):
    
    def create_coach_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            goal = specs.get('goal', 'Провести reflection сесію')
            norms = specs.get('norms', '')
            timebox = specs.get('timebox', '10 хвилин')
            asf_raw = specs.get('asf', '')
            constraints = specs.get('constraints', '')
            start_template = specs.get('start_template', '')
            summary_template = specs.get('summary_template', '')
            
            turn_count = len(ctx.state.answers)
            phase = ctx.state.custom_data.get('phase', 'aspiration')
            
            conversation_history = ""
            for i, ans in enumerate(ctx.state.answers[-10:]):
                conversation_history += f"\nTurn {i+1} [{ans.get('phase', 'unknown')}]:\n"
                conversation_history += f"Coachee: {ans.get('user_message', '')}\n"
                conversation_history += f"Coach: {ans.get('coach_response', '')}\n"
            
            if turn_count == 0:
                return f"""You are a reflection coach conducting an ASF session.

# Your Opening Message
{start_template}

# Session Goal
{goal}

# Norms
{norms}

# Timebox
{timebox}

# Constraints
{constraints}

IMPORTANT: Your first response MUST use the exact text from "Your Opening Message" above, then ask the first Aspiration question about self-attention. Max 120 words total."""
            
            asf_dict = {}
            if isinstance(asf_raw, str):
                try:
                    asf_dict = json.loads(asf_raw)
                except:
                    pass
            else:
                asf_dict = asf_raw
            
            aspiration_data = ctx.state.custom_data.get('aspiration', {})
            strengths_data = ctx.state.custom_data.get('strengths', {})
            feed_forward_data = ctx.state.custom_data.get('feed_forward', {})
            
            if phase == 'aspiration':
                aspiration_prompts = asf_dict.get('aspiration_questions', 'Запитайте про аспірацію')
                
                return f"""# Aspiration Phase

# Previous Conversation
{conversation_history}

# Goal
{goal}

# Aspiration Questions/Guide
{aspiration_prompts}

# Collected So Far
{json.dumps(aspiration_data, ensure_ascii=False, indent=2)}

# Your Task
- Ask ONE question at a time (max 120 words)
- After each response, provide a brief bullet summary (2-5 points)
- Push for specifics: dates, metrics, concrete examples
- DO NOT move to next phase until you have:
  * Clear aspiration statement
  * Time horizon
  * Core motivators with "why"

# Constraints
{constraints}

If you have collected enough for Aspiration (clear goal + timeline + motivation), signal by saying: "Чудово! Тепер перейдемо до ваших сильних сторін."
"""
            
            elif phase == 'strengths':
                strengths_prompts = asf_dict.get('strengths_questions', 'Запитайте про сильні сторони')
                
                return f"""# Strengths Phase

# Previous Conversation
{conversation_history}

# Goal
{goal}

# Aspiration (completed)
{json.dumps(aspiration_data, ensure_ascii=False, indent=2)}

# Strengths Questions/Guide
{strengths_prompts}

# Collected So Far
{json.dumps(strengths_data, ensure_ascii=False, indent=2)}

# Your Task
- Ask about their strengths, values, past successes
- Collect 1-2 concrete examples (STAR format if possible)
- Identify potential overuse risks
- Suggest concrete guardrails
- One question at a time (max 120 words)

# Constraints
{constraints}

If you have collected enough (2+ strengths with examples + overuse risks), signal by saying: "Відмінно! Тепер давайте визначимо конкретні наступні кроки."
"""
            
            elif phase == 'feed_forward':
                feed_forward_prompts = asf_dict.get('feed_forward_questions', 'Запитайте про наступні кроки')
                
                return f"""# Feed-forward Phase

# Previous Conversation
{conversation_history}

# Goal
{goal}

# Aspiration
{json.dumps(aspiration_data, ensure_ascii=False, indent=2)}

# Strengths
{json.dumps(strengths_data, ensure_ascii=False, indent=2)}

# Feed-forward Questions/Guide
{feed_forward_prompts}

# Collected So Far
{json.dumps(feed_forward_data, ensure_ascii=False, indent=2)}

# Your Task
- Help commit to ONE concrete next step
- Define: action + deadline + metric
- Create if-then plan (trigger → alternative)
- Set up accountability (person/event/reminder)
- Max 120 words

# Constraints
{constraints}

Once you have:
- One keystone action with deadline and metric
- If-then plan
- Accountability mechanism

Signal completion by saying: "Дякую за продуктивну сесію! Зараз підготую підсумок."
"""
            
            elif phase == 'summary':
                return f"""Prepare the final Reflection Canvas summary.

# Session Data
## Aspiration
{json.dumps(aspiration_data, ensure_ascii=False, indent=2)}

## Strengths
{json.dumps(strengths_data, ensure_ascii=False, indent=2)}

## Feed-forward
{json.dumps(feed_forward_data, ensure_ascii=False, indent=2)}

# Summary Template
{summary_template}

Fill the template with collected data. Be concise and structured."""
            
            else:
                return "Session complete."
        
        return Agent[WorkflowContext](
            name="ReflectionCoach",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, max_tokens=512)
        )
    
    async def run_workflow_stream(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> AsyncGenerator[str, None]:
        with trace(f"Reflection-{ub_id}"):
            specifications = self.parse_specifications(block)

            if specifications and not isinstance(specifications, list):
                specifications = [specifications]
            elif not specifications:
                specifications = []

            specs = specifications[0] if specifications else {}

            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)

            if state.status == "finished":
                yield "Reflection session завершено. Дякую!"
                return

            air_records = await xano.get_air_history(ub_id)
            state.answers = self._convert_air_to_history(air_records)

            context = WorkflowContext(state=state)

            if not state.custom_data.get('phase'):
                state.custom_data['phase'] = 'aspiration'
            if 'aspiration' not in state.custom_data:
                state.custom_data['aspiration'] = {}
            if 'strengths' not in state.custom_data:
                state.custom_data['strengths'] = {}
            if 'feed_forward' not in state.custom_data:
                state.custom_data['feed_forward'] = {}

            coach = self.create_coach_agent(context, specs, template.get("model", "gpt-4o"))
            result = Runner.run_streamed(coach, user_message, context=context)

            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    chunk = event.data.delta
                    full_response += chunk
                    yield chunk

            self._update_phase_and_data(state, user_message, full_response, specs)

            print(f"DEBUG reflection save: phase={state.custom_data.get('phase')}, answers={len(state.answers)}, custom_data_keys={list(state.custom_data.keys())}")
            await xano.save_workflow_state(state)

            if state.status == "finished":
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
    
    def _update_phase_and_data(self, state: WorkflowState, user_message: str, coach_response: str, specs: Dict):
        phase = state.custom_data.get('phase', 'aspiration')
        
        if phase == 'aspiration':
            aspiration = state.custom_data.get('aspiration', {})
            
            if user_message:
                aspiration['responses'] = aspiration.get('responses', [])
                aspiration['responses'].append(user_message)
            
            if 'перейдемо до' in coach_response.lower() and 'сильних сторін' in coach_response.lower():
                aspiration['completed'] = True
                state.custom_data['phase'] = 'strengths'
            
            state.custom_data['aspiration'] = aspiration
        
        elif phase == 'strengths':
            strengths = state.custom_data.get('strengths', {})
            
            if user_message:
                strengths['responses'] = strengths.get('responses', [])
                strengths['responses'].append(user_message)
            
            if 'наступні кроки' in coach_response.lower() or 'визначимо конкретні' in coach_response.lower():
                strengths['completed'] = True
                state.custom_data['phase'] = 'feed_forward'
            
            state.custom_data['strengths'] = strengths
        
        elif phase == 'feed_forward':
            feed_forward = state.custom_data.get('feed_forward', {})
            
            if user_message:
                feed_forward['responses'] = feed_forward.get('responses', [])
                feed_forward['responses'].append(user_message)
            
            if 'підсумок' in coach_response.lower() or 'продуктивну сесію' in coach_response.lower():
                feed_forward['completed'] = True
                state.custom_data['phase'] = 'summary'
            
            state.custom_data['feed_forward'] = feed_forward
        
        elif phase == 'summary':
            state.status = 'finished'
        
        timebox = specs.get('timebox', '')
        if '10' in timebox and len(state.answers) >= 8:
            if phase not in ['summary', 'finished']:
                state.custom_data['phase'] = 'summary'
        elif '20' in timebox and len(state.answers) >= 15:
            if phase not in ['summary', 'finished']:
                state.custom_data['phase'] = 'summary'
        
        if 'завершуй' in user_message.lower() or 'закінчи' in user_message.lower():
            state.custom_data['phase'] = 'summary'
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"ReflectionEval-{ub_id}"):
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
                    if crit.get('criterion_name'):
                        criteria_text += f": {crit['criterion_name']}"
                    criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
                    if crit.get('summary_instructions'):
                        criteria_text += f"Summary: {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"Grading: {crit['grading_instructions']}\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    phase = ans.get('phase', 'unknown')
                    conversation_text += f"\n### Turn {i+1} [{phase.upper()}]\n"
                    conversation_text += f"**Coachee:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Coach:** {ans.get('coach_response', 'N/A')}\n\n"
                
                aspiration_data = ctx.workflow_state.custom_data.get('aspiration', {})
                strengths_data = ctx.workflow_state.custom_data.get('strengths', {})
                feed_forward_data = ctx.workflow_state.custom_data.get('feed_forward', {})
                
                return f"""{ctx.eval_instructions}

# Reflection Session
{conversation_text}

# Collected Data
## Aspiration
{json.dumps(aspiration_data, ensure_ascii=False, indent=2)}

## Strengths
{json.dumps(strengths_data, ensure_ascii=False, indent=2)}

## Feed-forward
{json.dumps(feed_forward_data, ensure_ascii=False, indent=2)}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the reflection session based on ASF framework completeness and quality.

For each criterion:
1. Review the session conversation and collected data
2. Assess how well the criterion was met
3. Assign a grade (0 to max_points for that criterion)
4. Provide clear reasoning

Format your response as:

# Evaluation Report

## Criterion 1: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Why this grade was assigned]

## Criterion 2: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Explanation]

# Summary
**Total Score:** X/{total_max_points} points
**Overall Performance:** [Brief summary]
**Recommendations:** [Optional suggestions]"""
            
            agent = Agent[EvaluationContext](
                name="ReflectionEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text