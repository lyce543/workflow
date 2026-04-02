from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
import re

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace
from openai.types.responses import ResponseTextDeltaEvent

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class RoleplayWorkflow(BaseWorkflow):
    
    def create_roleplay_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            goal = specs.get('goal', '')
            role = specs.get('role', '')
            student_role = specs.get('student_role', '')
            behavior = specs.get('behavior', '')
            scenario = specs.get('basic_scenario', '')
            
            turn_count = len(ctx.state.answers)
            
            last_messages = ""
            if ctx.state.answers:
                last_messages = "\n# Recent Conversation (last 3 turns)\n"
                for ans in ctx.state.answers[-3:]:
                    last_messages += f"Student: {ans.get('user_message', '')[:100]}...\n"
                    last_messages += f"You: {ans.get('agent_response', '')[:100]}...\n\n"
            
            progress_notes = ctx.state.custom_data.get('progress_notes', [])
            progress_text = ""
            if progress_notes:
                progress_text = "\n# What has been accomplished:\n" + "\n".join(f"- {note}" for note in progress_notes[-5:])
            
            return f"""You are participating in a role-play simulation.

# Your Role
{role}

# Student's Role
{student_role}

# Learning Goal
{goal}

# Scenario Flow
{scenario}

# Behavior Rules
{behavior}

Current turn: {turn_count + 1}

{last_messages}

{progress_text}

IMPORTANT INSTRUCTIONS:
1. Stay in character at all times
2. Respond naturally to the student's actions and words
3. DO NOT repeat the same question - if student answered, move forward in the scenario
4. Follow the scenario flow step by step
5. Track progress and advance through the simulation
6. If student demonstrates understanding or completes a task, acknowledge it and move to the next part
7. Avoid circular conversations - each turn should make progress
8. Do not break the fourth wall

CRITICAL: If you notice you're asking similar questions repeatedly, STOP and move to the next stage of the scenario."""
        
        return Agent[WorkflowContext](
            name="RoleplayAgent",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.8, max_tokens=1024)
        )
    
    async def run_workflow_stream(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> AsyncGenerator[str, None]:
        with trace(f"Roleplay-{ub_id}"):
            specifications = self.parse_specifications(block)

            if specifications and not isinstance(specifications, list):
                specifications = [specifications]
            elif not specifications:
                specifications = []
            
            specs = specifications[0] if specifications else {}
            
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                yield "Role-play завершено. Дякую за участь!"
                return
            
            if not state.custom_data.get('progress_notes'):
                state.custom_data['progress_notes'] = []

            air_records = await xano.get_air_history(ub_id)
            state.answers = self._convert_air_to_history(air_records)

            context = WorkflowContext(state=state)

            agent = self.create_roleplay_agent(context, specs, template.get("model", "gpt-4o"))
            result = Runner.run_streamed(agent, user_message, context=context)
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    chunk = event.data.delta
                    full_response += chunk
                    yield chunk
            
            self._update_progress_tracking(state, user_message, full_response)
            
            finish_conditions = specs.get('finish_dialogue_conditions', '')
            should_finish = self._check_finish_conditions(state, finish_conditions, full_response)
            
            if should_finish:
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
            else:
                await xano.save_workflow_state(state)
    
    def _update_progress_tracking(self, state: WorkflowState, user_message: str, agent_response: str):
        keywords_progress = [
            'чудово', 'добре', 'правильно', 'згоден', 'зрозумів',
            'наступний', 'тепер', 'переходимо', 'good', 'great', 'correct', 'next'
        ]
        
        response_lower = agent_response.lower()
        if any(keyword in response_lower for keyword in keywords_progress):
            progress_notes = state.custom_data.get('progress_notes', [])
            progress_notes.append(f"Turn {len(state.answers)}: Progress made")
            state.custom_data['progress_notes'] = progress_notes[-10:]
    
    def _check_finish_conditions(self, state: WorkflowState, conditions: str, last_response: str) -> bool:
        if not conditions:
            return False
        
        turn_count = len(state.answers)
        
        if turn_count < 3:
            return False
        
        max_turns = 20
        turn_match = re.search(r'(\d+)[–\-\s]*(хвилин|turns?|exchanges?)', conditions.lower())
        if turn_match:
            number = int(turn_match.group(1))
            if 'хвилин' in turn_match.group(2):
                max_turns = max(number * 2, 10)
            else:
                max_turns = number
        
        if turn_count >= max_turns:
            return True
        
        completion_phrases = [
            'завершено', 'підсумок', 'дякую за сесію', 'це все',
            'completed', 'finished', 'that concludes', 'thank you for',
            'на цьому завершуємо', 'це завершує нашу розмову'
        ]
        
        response_lower = last_response.lower()
        if any(phrase in response_lower for phrase in completion_phrases):
            return True
        
        if 'finished' in conditions.lower() or 'phrase' in conditions.lower():
            conditions_lower = conditions.lower()
            if 'student' in conditions_lower and any(phrase in conditions_lower for phrase in ['підтверд', 'скаж', 'фраз']):
                last_answers = [ans.get('user_message', '').lower() for ans in state.answers[-3:]]
                
                confirmation_words = ['зрозумів', 'зрозуміла', 'дякую', 'так', 'готов', 'готова', 'finished', 'understood']
                if any(any(word in answer for word in confirmation_words) for answer in last_answers):
                    return True
        
        if turn_count >= 5:
            last_agent_responses = [ans.get('agent_response', '') for ans in state.answers[-3:]]
            
            if len(set(resp[:50] for resp in last_agent_responses)) <= 1:
                return True
        
        return False
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"RoleplayEval-{ub_id}"):
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
                    conversation_text += f"\n### Turn {ans.get('turn', i+1)}\n"
                    conversation_text += f"**Student:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Agent:** {ans.get('agent_response', 'N/A')}\n\n"
                
                return f"""{ctx.eval_instructions}

# Role-play Conversation
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Additional Context
Total turns: {len(ctx.workflow_state.answers)}
Status: {ctx.workflow_state.status}

# Your Task

Evaluate the student's performance in the role-play based on the criteria.
Consider both the quality of responses and whether the conversation progressed naturally without getting stuck.

For each criterion:
1. Review the role-play conversation
2. Assess how well the student met the criterion
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
                name="RoleplayEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text