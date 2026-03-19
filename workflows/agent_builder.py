from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
import httpx
import json

from openai import AsyncOpenAI

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext
from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace


class AgentBuilderWorkflow(BaseWorkflow):
    
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.api_key = openai_api_key
    
    async def _get_or_create_chatkit_session(self, workflow_id: str, user_id: str, state: WorkflowState) -> dict:
        existing_session = state.custom_data.get("chatkit_session")
        
        if existing_session:
            expires_at = existing_session.get("expires_at", 0)
            if datetime.now().timestamp() < expires_at:
                return existing_session
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chatkit/sessions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "chatkit_beta=v1"
                },
                json={
                    "workflow": {"id": workflow_id},
                    "user": user_id,
                    "expires_after": 1800,
                    "max_requests_per_session": 500
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to create ChatKit session: {response.status_code} - {response.text}")
            
            session_data = response.json()
            
            session_info = {
                "session_id": session_data.get("id"),
                "client_secret": session_data.get("client_secret"),
                "thread_id": session_data.get("thread_id"),
                "expires_at": datetime.now().timestamp() + 1800,
                "workflow_id": workflow_id
            }
            
            return session_info
    
    async def _send_message_to_chatkit(self, session_info: dict, user_message: str, user_files: list = []) -> AsyncGenerator[str, None]:
        client_secret = session_info.get("client_secret")

        if not client_secret:
            yield "Error: No client secret available."
            return

        content = []
        if user_message:
            content.append({"type": "input_text", "text": user_message})
        for f in user_files:
            file_data = f.get("file_data", "") or ""
            mime = f.get("type", "") or ""
            name = f.get("name", "file")
            url = f.get("url", "") or ""
            is_image = mime.startswith("image/") or file_data.startswith("data:image/")
            if is_image:
                content.append({"type": "input_image", "image_url": file_data or url})
            elif file_data:
                content.append({"type": "input_file", "filename": name, "file_data": file_data})
        if not content:
            content.append({"type": "input_text", "text": ""})

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chatkit/threads/messages",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {client_secret}",
                    "OpenAI-Beta": "chatkit_beta=v1",
                    "Accept": "text/event-stream"
                },
                json={
                    "content": content
                },
                timeout=120.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"Error: {response.status_code} - {error_text.decode()}"
                    return
                
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        
                        for line in event_str.split("\n"):
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    return
                                
                                try:
                                    event_data = json.loads(data)
                                    
                                    if event_data.get("type") == "assistant_message":
                                        content = event_data.get("content", [])
                                        for item in content:
                                            if item.get("type") == "output_text":
                                                text = item.get("text", "")
                                                if text:
                                                    yield text
                                    
                                    elif event_data.get("type") == "text_delta":
                                        delta = event_data.get("delta", "")
                                        if delta:
                                            yield delta
                                            
                                except json.JSONDecodeError:
                                    continue
    
    async def run_workflow_stream(
        self,
        block: Dict,
        template: Dict,
        user_message: str,
        ub_id: int,
        xano,
        user_files: List[Dict] = []
    ) -> AsyncGenerator[str, None]:
        workflow_id = block.get("workflow_id")
        
        if not workflow_id:
            yield "Error: No workflow_id configured for this block."
            return
        
        with trace(f"AgentBuilder-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                yield "Чат завершено."
                return
            
            try:
                session_info = await self._get_or_create_chatkit_session(
                    workflow_id=workflow_id,
                    user_id=str(ub_id),
                    state=state
                )
                state.custom_data["chatkit_session"] = session_info
                
            except Exception as e:
                print(f"Error creating ChatKit session: {e}")
                yield f"Error connecting to Agent Builder workflow: {str(e)}"
                return
            
            full_response = ""
            
            async for chunk in self._send_message_to_chatkit(session_info, user_message, user_files):
                full_response += chunk
                yield chunk
            
            state.answers.append({
                "user_message": user_message,
                "assistant_response": full_response,
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "session_id": session_info.get("session_id")
            })
            
            await xano.save_workflow_state(state)
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"AgentBuilderEval-{ub_id}"):
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
                        criteria_text += f"Summary Instructions: {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"Grading Instructions: {crit['grading_instructions']}\n"
                    criteria_text += "\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    conversation_text += f"\n{'='*60}\n"
                    conversation_text += f"Exchange {i+1}:\n"
                    conversation_text += f"{'='*60}\n\n"
                    conversation_text += f"**Student:** {ans.get('user_message', 'N/A')}\n\n"
                    conversation_text += f"**Assistant:** {ans.get('assistant_response', 'N/A')}\n\n"
                
                return f"""You are an evaluation assistant for an educational platform.

{ctx.eval_instructions}

# Conversation History
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the student's performance according to the provided criteria.

For each criterion:
1. Review the conversation
2. Assess how well they met the criterion
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
                name="AgentBuilderEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text
    
    def _calculate_total_points(self, criteria: List[Dict[str, Any]]) -> int:
        total = 0
        for crit in criteria:
            total += crit.get('max_points', 0)
        return total
    
    def parse_specifications(self, block: Dict) -> List[Dict]:
        specs = block.get("int_specification_json", [])
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except:
                specs = []
        return specs if isinstance(specs, list) else [specs]