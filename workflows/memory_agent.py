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


class MemoryAgentWorkflow(BaseWorkflow):

    def _build_mcp_tool(self) -> HostedMCPTool:
        return HostedMCPTool(tool_config={
            "type": "mcp",
            "server_label": "user_data",
            "allowed_tools": ["get_user_data", "write_user_data"],
            "require_approval": "never",
            "server_url": MCP_SERVER_URL
        })

    def _create_agent(self, context: MemoryAgentContext, model: str) -> Agent:
        mcp = self._build_mcp_tool()

        def agent_instructions(run_context: RunContextWrapper[MemoryAgentContext], _agent: Agent):
            ctx = run_context.context

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
{ctx.agent_specifications}"""

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

    async def _process_files(self, files: List[Dict], client: httpx.AsyncClient) -> List[Dict]:
        content = []
        for f in files:
            url = f.get("url", "")
            mime = f.get("type", "")
            name = f.get("name", "file")
            file_data = f.get("file_data", "")
            is_image = mime.startswith("image/") or file_data.startswith("data:image/")
            if is_image:
                image_url = file_data or url
                if image_url:
                    content.append({"type": "input_image", "image_url": image_url})
            else:
                if file_data:
                    content.append({"type": "input_file", "filename": name, "file_data": file_data})
                elif url:
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        encoded = base64.b64encode(resp.content).decode("utf-8")
                        content.append({"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{encoded}"})
                    except Exception as e:
                        print(f"Could not fetch file {name}: {e}")
        return content

    async def _build_full_input(self, conversation_history: List[Dict], user_message: str, user_files: List[Dict]) -> Any:
        messages = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for turn in conversation_history:
                user_content = []
                if turn.get("user_message"):
                    user_content.append({"type": "input_text", "text": turn["user_message"]})
                user_content += await self._process_files(turn.get("user_files", []), client)
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                if turn.get("agent_response"):
                    messages.append({"role": "assistant", "content": turn["agent_response"]})

            current_content = []
            if user_message:
                current_content.append({"type": "input_text", "text": user_message})
            current_content += await self._process_files(user_files, client)
            if current_content:
                messages.append({"role": "user", "content": current_content})

        return messages if messages else user_message

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
            )

            model = template.get("model", "gpt-4o")
            agent = self._create_agent(context, model)

            agent_input = await self._build_full_input(state.answers, user_message, user_files)
            result = Runner.run_streamed(agent, agent_input, context=context)

            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    chunk = event.data.delta
                    full_response += chunk
                    yield chunk

            state.answers.append({
                "turn": len(state.answers) + 1,
                "user_message": user_message,
                "user_files": user_files,
                "agent_response": full_response,
                "timestamp": datetime.now().isoformat()
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
        return ""