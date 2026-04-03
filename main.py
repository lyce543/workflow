import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from models import StudentMessage, AssistantResponse, ChatStatus
from xano_client import XanoClient
from workflows import get_workflow_class, MemoryAgentWorkflow, AgentBuilderWorkflow
from fastapi import UploadFile, File

import tiktoken

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text) // 4


app = FastAPI(title="EdTech AI Platform", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.alsie.app",
        "https://alsie.app",
        "https://alsie-app.webflow.io",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY, Config.OPENAI_API_KEY)

chatkit_server = None

def get_chatkit_server():
    global chatkit_server
    if chatkit_server is None:
        from chatkit_server import AlsieChatKitServer
        chatkit_server = AlsieChatKitServer(Config.OPENAI_API_KEY, xano)
    return chatkit_server


@app.get("/")
async def root():
    return {"status": "operational", "version": "5.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xano_configured": bool(Config.XANO_BASE_URL),
        "openai_configured": bool(Config.OPENAI_API_KEY)
    }


@app.options("/chat/message")
async def chat_message_options():
    return {"status": "ok"}


@app.post("/chat/message")
async def process_student_message(message: StudentMessage):
    try:
        print(f"=== START: Processing message for ub_id: {message.ub_id} ===")
        
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        if session.get("status") == "idle":
            print(f"Updating status from idle to started for ub_id: {message.ub_id}")
            await xano.update_chat_status(message.ub_id, status=ChatStatus.STARTED)
        
        workflow_id = block.get("workflow_id")

        if workflow_id and workflow_id != 'self-hosted':
            print(f"OpenAI-hosted workflow: {workflow_id}")
            workflow = AgentBuilderWorkflow(Config.OPENAI_API_KEY)
        else:
            template_id = block["int_template_id"]
            print(f"Template ID: {template_id}")
            workflow_class = get_workflow_class(template_id)

            if not workflow_class:
                raise HTTPException(status_code=400, detail=f"No workflow found for template {template_id}")

            print(f"Workflow class: {workflow_class.__name__}")
            workflow = workflow_class(Config.OPENAI_API_KEY)
        
        course_id = block.get("_lesson", {}).get("course_id") or block.get("_lesson", {}).get("_course", {}).get("id") or session.get("course_id") or 0
        user_id = session.get("user_id") or 0
        block_id = block.get("id") or session.get("block_id")
        model = template_data.get("model", "gpt-4o")
        
        async def generate():
            full_response = ""
            print(f"Starting stream for ub_id: {message.ub_id}")
            chunk_count = 0

            # Зберігаємо повідомлення студента в air
            user_files_for_air = [
                {"url": f.get("url", ""), "type": f.get("type", ""), "name": f.get("name", "")}
                for f in (message.files or [])
            ]
            air_record = await xano.add_air_message(
                ub_id=message.ub_id,
                user_id=user_id,
                block_id=block_id,
                text=message.content,
                user_files=user_files_for_air
            )
            air_id = air_record.get("id")

            if message.files and isinstance(workflow, (MemoryAgentWorkflow, AgentBuilderWorkflow)):
                stream = workflow.run_workflow_stream(block, template_data, message.content, message.ub_id, xano, user_files=message.files)
            else:
                stream = workflow.run_workflow_stream(block, template_data, message.content, message.ub_id, xano)
            async for chunk in stream:
                chunk_count += 1
                print(f"Chunk {chunk_count}: {chunk[:50]}..." if len(chunk) > 50 else f"Chunk {chunk_count}: {chunk}")
                full_response += chunk
                yield chunk

            print(f"Stream complete. Total chunks: {chunk_count}")
            print(f"Full response length: {len(full_response)} characters")

            # Оновлюємо запис в air з відповіддю AI
            if air_id:
                await xano.update_air_message(
                    air_id=air_id,
                    ai_content=[{"text": full_response}],
                    status="completed"
                )

            input_tokens = estimate_tokens(message.content, model)
            output_tokens = estimate_tokens(full_response, model)

            await xano.save_token_usage(
                ub_id=message.ub_id,
                block_id=block_id,
                course_id=course_id,
                user_id=user_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                operation_type="chat"
            )

            print(f"Token usage saved: input={input_tokens}, output={output_tokens}")
            print(f"=== END: Message processing for ub_id: {message.ub_id} ===\n")
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing message: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)

        if session.get('work_summary'):
            return {
                "evaluation": session['work_summary'],
                "timestamp": datetime.now().isoformat(),
                "conversation_length": 0,
                "criteria_count": 0,
                "cached": True
            }
        
        block = await xano.get_block(session["block_id"])
        
        eval_instructions = block.get("eval_instructions")
        if not eval_instructions:
            raise HTTPException(status_code=400, detail="No evaluation instructions configured")
        
        workflow_state = await xano.get_workflow_state(ub_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="No workflow state found")

        # Підставляємо історію з air для евалуейшну
        air_records = await xano.get_air_history(ub_id)
        if air_records:
            import json as _json

            def _parse_field(value, fallback):
                if isinstance(value, str):
                    try:
                        return _json.loads(value)
                    except Exception:
                        pass
                return value if value else fallback

            def _user_text(r):
                uc = _parse_field(r.get("user_content"), {})
                return uc.get("text", "") if isinstance(uc, dict) else ""

            def _ai_text(r):
                ac = _parse_field(r.get("ai_content"), [])
                if isinstance(ac, list) and ac:
                    return ac[0].get("text", "") if isinstance(ac[0], dict) else ""
                return ""

            workflow_state.answers = [
                {
                    "user_message": _user_text(r),
                    "agent_response": _ai_text(r),
                    "interviewer_question": _ai_text(r),
                    "answer": _user_text(r),
                    "evaluation": {}
                }
                for r in air_records
            ]

        import json
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []

        print(f"DEBUG block keys: {list(block.keys())}")
        print(f"DEBUG eval_crit_json raw: {block.get('eval_crit_json')}")
        print(f"DEBUG eval criteria for ub_id={ub_id}: {criteria}")

        template_id = block["int_template_id"]
        workflow_class = get_workflow_class(template_id)
        
        if not workflow_class:
            raise HTTPException(status_code=400, detail=f"No workflow found for template {template_id}")
        
        workflow = workflow_class(Config.OPENAI_API_KEY)
        
        evaluation_text = await workflow.run_evaluation(
            ub_id=ub_id,
            workflow_state=workflow_state,
            eval_instructions=eval_instructions,
            criteria=criteria,
            model=block.get("model", "gpt-4o")
        )
        
        print(f"Saving evaluation to Xano via update_ub endpoint...")
        
        update_result = await xano.update_chat_status(ub_id, grade=evaluation_text)
        
        if update_result:
            print(f"Grade saved successfully: {update_result}")
        else:
            print(f"Grade save returned empty result")
        
        return {
            "evaluation": evaluation_text,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(workflow_state.answers),
            "criteria_count": len(criteria),
            "cached": False,
            "grade_saved": bool(update_result)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lesson/{lesson_id}/evaluate-tests")
async def evaluate_lesson_tests(lesson_id: int):
    try:
        response = await xano.client.get(
            f"{xano.base_url}/test_ub",
            params={"lesson_id": lesson_id}
        )
        if not response.is_success:
            raise HTTPException(status_code=400, detail="Failed to fetch test blocks")

        blocks_data = response.json().get("progress_by_module", [])

        results = []
        for block in blocks_data:
            for test in block.get("tests", []):
                if test.get("status") in ["finished", "started"]:
                    ub_id = test["id"]
                    try:
                        result = await evaluate_chat(ub_id)
                        results.append({"ub_id": ub_id, "status": "ok"})
                    except Exception as e:
                        results.append({"ub_id": ub_id, "status": "error", "error": str(e)})

        return {"evaluated": len(results), "results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/state")
async def get_chat_state(ub_id: int, offset: int = 0, limit: int = 10):
    try:
        import json as _json

        workflow_state = await xano.get_workflow_state(ub_id)

        if not workflow_state:
            raise HTTPException(status_code=404, detail="No workflow state found")

        air_records = await xano.get_air_history(ub_id)

        if air_records:
            def _parse(value, fallback):
                if isinstance(value, str):
                    try:
                        return _json.loads(value)
                    except Exception:
                        pass
                return value if value is not None else fallback

            all_answers = []
            for r in air_records:
                uc = _parse(r.get("user_content"), {})
                ac = _parse(r.get("ai_content"), [])
                uf = _parse(r.get("user_files"), [])
                user_text = uc.get("text", "") if isinstance(uc, dict) else ""
                ai_text = ac[0].get("text", "") if isinstance(ac, list) and ac else ""
                all_answers.append({
                    "user_message": user_text,
                    "agent_response": ai_text,
                    "user_files": uf if isinstance(uf, list) else [],
                    "timestamp": r.get("created_at", "")
                })
        else:
            all_answers = workflow_state.answers

        total = len(all_answers)
        # повертаємо з кінця: offset=0 → останні limit повідомлень
        start = max(0, total - limit - offset)
        end = max(0, total - offset)
        answers = all_answers[start:end]

        return {
            "ub_id": workflow_state.ub_id,
            "block_id": workflow_state.block_id,
            "current_question_index": workflow_state.current_question_index,
            "questions": workflow_state.questions,
            "answers": answers,
            "total": total,
            "has_more": start > 0,
            "follow_up_count": workflow_state.follow_up_count,
            "max_follow_ups": workflow_state.max_follow_ups,
            "status": workflow_state.status,
            "custom_data": workflow_state.custom_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting chat state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{ub_id}/state")
async def delete_chat_state(ub_id: int):
    try:
        success = await xano.delete_workflow_state(ub_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow state not found or could not be deleted")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting chat state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AddFilesRequest(BaseModel):
    files: list


@app.post("/chat/{ub_id}/add_files")
async def add_files_to_last_answer(ub_id: int, request: AddFilesRequest):
    # Files are now stored directly in the air table record via add_air_message.
    # This endpoint is kept for backwards compatibility but is a no-op.
    return {"success": True}


class ChatKitSessionRequest(BaseModel):
    workflow_id: str
    user_id: str = "anonymous"
    ub_id: str | None = None


@app.post("/chatkit/session")
async def create_chatkit_session(request: ChatKitSessionRequest):
    try:
        from openai import OpenAI

        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        workflow_params = {"id": request.workflow_id}
        if request.ub_id:
            workflow_params["state_variables"] = {"ub_id": request.ub_id}

        session = client.beta.chatkit.sessions.create(
            user=request.user_id,
            workflow=workflow_params,
            expires_after={"anchor": "created_at", "seconds": 600}
        )

        thread_id = None
        try:
            threads = client.beta.chatkit.threads.list(user=request.user_id, limit=1, order="desc")
            if threads.data:
                thread_id = threads.data[0].id
        except Exception as e:
            print(f"Could not fetch existing thread: {e}")

        return {
            "client_secret": session.client_secret,
            "session_id": session.id,
            "expires_at": session.expires_at,
            "thread_id": thread_id
        }

    except Exception as e:
        print(f"Error creating ChatKit session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chatkit/thread/agent/{user_id}")
async def get_agent_builder_thread(user_id: str):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        threads = client.beta.chatkit.threads.list(user=user_id, limit=1, order="desc")
        thread_id = threads.data[0].id if threads.data else None
        return {"thread_id": thread_id}
    except Exception as e:
        print(f"Error fetching agent builder thread: {e}")
        return {"thread_id": None}


@app.post("/chatkit/upload")
async def chatkit_upload(request: Request, file: UploadFile = File(...)):
    try:
        ub_id = request.query_params.get("ub_id")
        block_id = request.query_params.get("block_id")
        
        contents = await file.read()
        file_id = f"file_{ub_id}_{int(datetime.now().timestamp() * 1000)}"
        
        server = get_chatkit_server()
        
        await server.file_store.save_file(
            file_id=file_id,
            content=contents,
            metadata={
                "name": file.filename,
                "mime_type": file.content_type,
                "size": len(contents),
                "ub_id": ub_id,
                "block_id": block_id
            }
        )
        
        from chatkit.types import FileAttachment
        from chatkit_server import RequestContext
        
        attachment = FileAttachment(
            id=file_id,
            name=file.filename,
            mime_type=file.content_type or "application/octet-stream",
        )
        
        context = RequestContext(
            user_id="system",
            ub_id=int(ub_id) if ub_id else None,
            block_id=int(block_id) if block_id else None,
        )
        
        await server.store.save_attachment(attachment, context)
        
        return {
            "id": file_id,
            "name": file.filename,
            "mime_type": file.content_type or "application/octet-stream",
            "type": "file"
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    try:
        from chatkit.server import StreamingResult
        from chatkit_server import RequestContext
        
        body = await request.body()
        
        ub_id = request.query_params.get("ub_id")
        block_id = request.query_params.get("block_id")
        user_id = request.query_params.get("user_id", "anonymous")
        
        context = RequestContext(
            user_id=user_id,
            ub_id=int(ub_id) if ub_id else None,
            block_id=int(block_id) if block_id else None,
        )
        
        server = get_chatkit_server()
        result = await server.process(body, context)
        
        if isinstance(result, StreamingResult):
            return StreamingResponse(result, media_type="text/event-stream")
        return Response(content=result.json, media_type="application/json")
        
    except Exception as e:
        print(f"ChatKit endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _extract_turn(ans: dict):
    """Extract (user_text, ai_text, created_at_ms) from any old answer format."""
    user_text = (
        ans.get('user_message') or
        ans.get('answer') or
        ''
    ).strip()
    ai_text = (
        ans.get('agent_response') or
        ans.get('coach_response') or
        ans.get('assistant_response') or
        ans.get('tutor_response') or
        ans.get('interviewer_question') or
        ''
    ).strip()

    # Convert original ISO timestamp to milliseconds
    created_at_ms = None
    ts = ans.get('timestamp')
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            created_at_ms = int(dt.timestamp() * 1000)
        except Exception:
            pass

    return user_text, ai_text, created_at_ms


@app.get("/admin/migrate-to-air/preview")
async def preview_migration():
    """Dry-run: shows what would be migrated without making any changes."""
    states = await xano.get_all_workflow_states_with_answers()
    preview = []

    for state in states:
        ub_id = state.get("ub_id")
        if not ub_id:
            continue
        existing = await xano.get_air_history(ub_id)
        answers = state.get("_parsed_answers", [])
        turns = []
        for ans in answers:
            user_text, ai_text, created_at_ms = _extract_turn(ans)
            if user_text or ai_text:
                turns.append({
                    "user": user_text[:80] + "..." if len(user_text) > 80 else user_text,
                    "ai": ai_text[:80] + "..." if len(ai_text) > 80 else ai_text,
                    "has_files": bool(ans.get('user_files')),
                    "timestamp": ans.get('timestamp', '')
                })
        preview.append({
            "ub_id": ub_id,
            "block_id": state.get("block_id"),
            "turns_count": len(turns),
            "already_in_air": len(existing),
            "will_migrate": len(existing) == 0,
            "turns_preview": turns[:3]
        })

    total_to_migrate = sum(1 for p in preview if p["will_migrate"])
    total_skip = sum(1 for p in preview if not p["will_migrate"])
    return {
        "total_chats": len(preview),
        "will_migrate": total_to_migrate,
        "will_skip": total_skip,
        "chats": preview
    }


@app.post("/admin/migrate-to-air")
async def migrate_workflow_answers_to_air():
    """
    One-time migration: copies all workflow_state.answers to the air table.
    Safe to run multiple times — skips ub_ids that already have air records.
    Preserves original message timestamps.
    """
    states = await xano.get_all_workflow_states_with_answers()

    if not states:
        return {"migrated": 0, "skipped": 0, "message": "No workflow states with answers found"}

    migrated = 0
    skipped = 0
    errors = []

    for state in states:
        ub_id = state.get("ub_id")
        if not ub_id:
            continue

        try:
            existing = await xano.get_air_history(ub_id)
            if existing:
                skipped += 1
                continue

            answers = state.get("_parsed_answers", [])
            block_id = state.get("block_id") or 0

            for ans in answers:
                user_text, ai_text, created_at_ms = _extract_turn(ans)

                if not user_text and not ai_text:
                    continue

                user_files = ans.get('user_files') or []
                # Normalize file format
                if user_files and isinstance(user_files[0], dict):
                    user_files = [
                        {"url": f.get("url", ""), "type": f.get("type", ""), "name": f.get("name", "")}
                        for f in user_files
                    ]

                air_record = await xano.add_air_message(
                    ub_id=ub_id,
                    user_id=0,
                    block_id=block_id,
                    text=user_text,
                    user_files=user_files,
                    created_at=created_at_ms
                )
                air_id = air_record.get("id")

                if air_id and ai_text:
                    await xano.update_air_message(
                        air_id=air_id,
                        ai_content=[{"text": ai_text}],
                        status="completed"
                    )

            migrated += 1

        except Exception as e:
            errors.append({"ub_id": ub_id, "error": str(e)})

    return {
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors,
        "message": f"Migration complete: {migrated} migrated, {skipped} already had air records"
    }


@app.get("/usage/course/{course_id}")
async def get_course_usage(course_id: int):
    try:
        usage = await xano.get_course_token_usage(course_id)
        return usage
    except Exception as e:
        print(f"Error getting course usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage/course/{course_id}/by_block")
async def get_course_usage_by_block(course_id: int):
    try:
        usage = await xano.get_course_token_usage_by_block(course_id)
        return usage
    except Exception as e:
        print(f"Error getting course usage by block: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage/course/{course_id}/user/{user_id}")
async def get_user_usage(course_id: int, user_id: int):
    try:
        usage = await xano.get_user_token_usage(course_id, user_id)
        return usage
    except Exception as e:
        print(f"Error getting user usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage/course/{course_id}/period")
async def get_course_usage_by_period(course_id: int, start_date: str, end_date: str):
    try:
        usage = await xano.get_course_token_usage_by_period(course_id, start_date, end_date)
        return usage
    except Exception as e:
        print(f"Error getting course usage by period: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/chatkit/threads")
async def get_chatkit_threads(ub_id: int):
    try:
        from chatkit_server import RequestContext
        
        context = RequestContext(
            user_id=f"user_{ub_id}",
            ub_id=ub_id,
            block_id=None
        )
        
        server = get_chatkit_server()
        
        threads_page = await server.store.load_threads(
            limit=10,
            after=None,
            order="desc",
            context=context
        )
        
        threads = [
            {
                "id": thread.id,
                "created_at": thread.created_at.isoformat() if hasattr(thread.created_at, 'isoformat') else str(thread.created_at)
            }
            for thread in threads_page.data
        ]
        
        return {"threads": threads, "has_more": threads_page.has_more}
        
    except Exception as e:
        print(f"Error getting ChatKit threads: {e}")
        return {"threads": [], "has_more": False}

@app.get("/chatkit/config/{ub_id}")
async def get_chatkit_config(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        return {
            "allow_multiple_chats": template_data.get("allow_multiple_chats", True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/chatkit/thread/{ub_id}")
async def get_chatkit_thread_id(ub_id: int):
    workflow_state = await xano.get_workflow_state(ub_id)
    if workflow_state and workflow_state.custom_data:
        thread_id = workflow_state.custom_data.get('chatkit_thread_id')
        if thread_id:
            return {"thread_id": thread_id}
    return {"thread_id": None}
    
@app.get("/lesson/{lesson_id}/export-grades")
async def export_lesson_grades(lesson_id: int):
    try:
        import csv
        from io import StringIO
        import httpx
        
        url = f"{Config.XANO_BASE_URL}/get_progress_by_lesson?lesson_id={lesson_id}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            
            if not response.is_success:
                raise HTTPException(status_code=400, detail=f"Failed to fetch lesson progress: {response.text}")
            
            result = response.json()
            students_data = result.get('progress_by_module', [])
        
        if not students_data or len(students_data) == 0:
            raise HTTPException(status_code=404, detail="No student data found for this lesson")
        
        output = StringIO()
        writer = csv.writer(output)
        
        headers = ['Student Name', 'Student Email', 'Block Name', 'Status', 'Grade', 'Max Points', 'Summary', 'Comment']
        writer.writerow(headers)
        
        for student in students_data:
            student_name = student.get('student_name', '')
            student_email = student.get('student_email', '')
            
            for block in student.get('blocks', []):
                block_name = block.get('block_name', '')
                status = block.get('status', '')
                grading_output = block.get('grading_output')
                
                if grading_output and isinstance(grading_output, list) and len(grading_output) > 0:
                    for criterion in grading_output:
                        grade = criterion.get('grade', '')
                        max_points = criterion.get('max_points', '')
                        summary = criterion.get('summary', '')
                        comment = criterion.get('grading_comment', '')
                        criterion_name = criterion.get('criterion_name', '')
                        
                        display_block_name = f"{block_name} - {criterion_name}" if criterion_name else block_name
                        
                        row = [student_name, student_email, display_block_name, status, grade, max_points, summary, comment]
                        writer.writerow(row)
                else:
                    row = [student_name, student_email, block_name, status, '', '', '', '']
                    writer.writerow(row)
        
        output.seek(0)
        
        from fastapi.responses import Response
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=lesson_{lesson_id}_grades.csv"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Export error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))