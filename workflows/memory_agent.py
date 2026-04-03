from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
import base64
import httpx

from agents import Agent, Runner, ModelSettings, RunContextWrapper, HostedMCPTool, trace
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.shared.reasoning import Reasoning

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


MCP_SERVER_URL = "https://xxye-mqg7-lvux.n7d.xano.io/x2/mcp/Kla8XVg_/mcp/stream"


class MemoryAgentContext:
    def __init__(
        self,
        course_id: int,
        lesson_id: int,
        block_id: int,
        user_id: int,
        level: str,
        include_all_children: bool,
        reading_instructions: str,
        writing_instructions: str,
        agent_instructions: str,
        agent_specifications: str,
        conversation_history: List[Dict[str, Any]]
    ):
        self.course_id = course_id
        self.lesson_id = lesson_id
        self.block_id = block_id
        self.user_id = user_id
        self.level = level
        self.include_all_children = include_all_children
        self.reading_instructions = reading_instructions
        self.writing_instructions = writing_instructions
        self.agent_instructions = agent_instructions
        self.agent_specifications = agent_specifications
        self.conversation_history = conversation_history


class MemoryAgentWorkflow(BaseWorkflow):

    def _build_mcp_tool(self) -> HostedMCPTool:
        return HostedMCPTool(tool_config={
            "type": "mcp",
            "server_label": "user_data",
            "require_approval": "never",
            "server_url": MCP_SERVER_URL
        })

    def _create_agent(self, context: MemoryAgentContext, model: str) -> Agent:
        mcp = self._build_mcp_tool()

        def agent_instructions(run_context: RunContextWrapper[MemoryAgentContext], _agent: Agent):
            ctx = run_context.context

            conversation_history = ""
            if ctx.conversation_history:
                conversation_history = "\n\n# CONVERSATION HISTORY:\n"
                for i, turn in enumerate(ctx.conversation_history, 1):
                    conversation_history += f"\nTurn {i}:\n"
                    conversation_history += f"Student: {turn.get('user_message', '')}\n"
                    conversation_history += f"You: {turn.get('agent_response', '')}\n"

            return f"""Use these inputs when calling the MCP server:
course_id: {ctx.course_id}
lesson_id: {ctx.lesson_id}
block_id: {ctx.block_id}
user_id: {ctx.user_id}
level: {ctx.level}
include_all_children: {ctx.include_all_children}

# Reading user data
{ctx.reading_instructions}

# Writing user data
{ctx.writing_instructions}

# Instructions
{ctx.agent_instructions}

# Specifications
{ctx.agent_specifications}

{conversation_history}"""

        return Agent[MemoryAgentContext](
            name="MemoryAgent",
            instructions=agent_instructions,
            model=model,
            tools=[mcp],
            model_settings=ModelSettings(
                store=True,
                reasoning=Reasoning(effort="medium", summary="auto")
            )
        )

    async def _build_input(self, user_message: str, user_files: List[Dict], conversation_history: List[Dict] = []) -> Any:
        content = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Add files from previous turns
            for turn in conversation_history:
                for f in turn.get("user_files", []):
                    file_data = f.get("file_data", "") or ""
                    mime = f.get("type", "") or ""
                    url = f.get("url", "") or ""
                    name = f.get("name", "file")
                    turn_label = turn.get("turn", "?")
                    is_image = mime.startswith("image/") or file_data.startswith("data:image/")
                    if is_image:
                        image_url = file_data or url
                        if image_url:
                            content.append({"type": "input_text", "text": f"[Image from turn {turn_label}:]"})
                            content.append({"type": "input_image", "image_url": image_url})
                    else:
                        if file_data:
                            content.append({"type": "input_text", "text": f"[File from turn {turn_label}: {name}]"})
                            content.append({"type": "input_file", "filename": name, "file_data": file_data})
                        elif url:
                            try:
                                resp = await client.get(url)
                                resp.raise_for_status()
                                encoded = base64.b64encode(resp.content).decode("utf-8")
                                content.append({"type": "input_text", "text": f"[File from turn {turn_label}: {name}]"})
                                content.append({"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{encoded}"})
                            except Exception as e:
                                print(f"Could not fetch historical file {name}: {e}")

            # Add current message text
            if user_message:
                content.append({"type": "input_text", "text": user_message})

            # Add current files
            for f in user_files:
                url = f.get("url", "") or ""
                mime = f.get("type", "") or ""
                name = f.get("name", "file")
                file_data = f.get("file_data", "") or ""
                is_image = mime.startswith("image/") or file_data.startswith("data:image/")
                if is_image:
                    content.append({"type": "input_image", "image_url": file_data or url})
                else:
                    if file_data:
                        content.append({"type": "input_file", "filename": name, "file_data": file_data})
                    else:
                        try:
                            resp = await client.get(url)
                            resp.raise_for_status()
                            encoded = base64.b64encode(resp.content).decode("utf-8")
                            content.append({"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{encoded}"})
                        except Exception as e:
                            print(f"Could not fetch file {name}: {e}")

        if len(content) > 1:
            return [{"role": "user", "content": content}]
        return user_message

    async def run_workflow_stream(
        self,
        block: Dict,
        template: Dict,
        user_message: str,
        ub_id: int,
        xano,
        user_files: List[Dict] = []
    ) -> AsyncGenerator[str, None]:

        with trace(f"MemoryAgent-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)

            if state.status == "finished":
                yield "Чат завершено."
                return

            specs = specifications[0] if specifications else {}

            course_id = (
                block.get("_lesson", {}).get("_course", {}).get("id")
                or block.get("_lesson", {}).get("course_id")
                or 0
            )
            lesson_id = block.get("_lesson", {}).get("id") or block.get("lesson_id") or 0
            block_id = block.get("id") or 0

            session = await xano.get_chat_session(ub_id)
            user_id = session.get("user_id") or 0

            air_records = await xano.get_air_history(ub_id)
            print(f"DEBUG memory_agent: air_records count={len(air_records)}")
            conversation_history = self._convert_air_to_history(air_records)
            print(f"DEBUG memory_agent: conversation_history turns={len(conversation_history)}, first_user_msg={conversation_history[0].get('user_message','')[:50] if conversation_history else 'EMPTY'}")

            context = MemoryAgentContext(
                course_id=course_id,
                lesson_id=lesson_id,
                block_id=block_id,
                user_id=user_id,
                level=specs.get("level", "course"),
                include_all_children=specs.get("include_all_children", True),
                reading_instructions=specs.get("reading_user_data_instructions", "Call MCP server to get user data."),
                writing_instructions=specs.get("writing_user_data_instructions", "When the user mentions a fact about themselves, write it on course level."),
                agent_instructions=block.get("int_instructions", "You are a helpful agent."),
                agent_specifications=specs.get("agent_specifications", ""),
                conversation_history=conversation_history
            )

            model = template.get("model", "gpt-4o")
            agent = self._create_agent(context, model)

            agent_input = await self._build_input(user_message, user_files, conversation_history)
            result = Runner.run_streamed(agent, agent_input, context=context)

            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    chunk = event.data.delta
                    full_response += chunk
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
        with trace(f"MemoryAgentEval-{ub_id}"):
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
                    criteria_text += "\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    conversation_text += f"\n{'='*60}\n"
                    conversation_text += f"Exchange {i+1}:\n"
                    conversation_text += f"{'='*60}\n\n"
                    conversation_text += f"**User:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Agent:** {ans.get('agent_response', 'N/A')}\n\n"

                return f"""{ctx.eval_instructions}

# Conversation History
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the conversation based on the criteria provided.

For each criterion:
1. Review the conversation exchanges
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
                name="MemoryAgentEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )

            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)

            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()

            return evaluation_text