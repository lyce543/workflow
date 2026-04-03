import httpx
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
from openai import OpenAI

from workflows.base import WorkflowState
from models import ChatStatus


class CriterionGrade(BaseModel):
    criterion_name: str
    grade: float
    max_points: float
    summary: str
    grading_comment: str

class EvaluationParsed(BaseModel):
    criteria: List[CriterionGrade]
    total_score: float


class XanoClient:
    def __init__(self, base_url: str, api_key: str, openai_api_key: str = None):
        self.base_url = base_url.rstrip('/')
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
        self.openai_api_key = openai_api_key
    
    async def get_block(self, block_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/block/{block_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_template(self, template_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/template/{template_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_chat_session(self, ub_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/ub/{ub_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_workflow_state(self, ub_id: int) -> Optional[WorkflowState]:
        try:
            response = await self.client.get(f"{self.base_url}/get_workflow_state/{ub_id}")
            if response.status_code == 200:
                data = response.json()
                if data and not data.get('error'):
                    data['questions'] = json.loads(data['questions']) if isinstance(data.get('questions'), str) else data.get('questions', [])
                    raw_ans = data.get('answers')
                    if isinstance(raw_ans, str):
                        try:
                            data['answers'] = json.loads(raw_ans)
                        except Exception:
                            data['answers'] = []
                    elif isinstance(raw_ans, list):
                        data['answers'] = raw_ans
                    else:
                        data['answers'] = []
                    raw_cd = data.get('custom_data')
                    if isinstance(raw_cd, str) and raw_cd.strip():
                        try:
                            data['custom_data'] = json.loads(raw_cd)
                        except Exception:
                            data['custom_data'] = {}
                    elif isinstance(raw_cd, dict):
                        data['custom_data'] = raw_cd
                    else:
                        data['custom_data'] = {}
                    return WorkflowState(**data)
        except Exception as e:
            print(f"Error loading workflow state: {e}")
        return None
    
    async def delete_workflow_state(self, ub_id: int):
        response = await self.client.delete(f"{self.base_url}/workflow_state/{ub_id}")
        return response.status_code in [200, 204]

    async def save_workflow_state(self, state: WorkflowState):
        data = {
            "ub_id": state.ub_id,
            "block_id": state.block_id,
            "current_question_index": state.current_question_index,
            "questions": json.dumps(state.questions, ensure_ascii=False),
            "follow_up_count": state.follow_up_count,
            "max_follow_ups": state.max_follow_ups,
            "status": state.status,
            "custom_data": json.dumps(state.custom_data, ensure_ascii=False)
        }
        response = await self.client.post(f"{self.base_url}/save_workflow_state", json=data)
        return response.json() if response.status_code in [200, 201] else None
    
    async def get_air_history(self, ub_id: int) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get(f"{self.base_url}/air/history", params={"ub_id": ub_id})
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("items", data.get("data", []))
        except Exception as e:
            print(f"Error loading air history: {e}")
        return []

    async def add_air_message(self, ub_id: int, user_id: int, block_id: int, text: str, user_files: List[Dict] = [], created_at: Optional[int] = None) -> Dict[str, Any]:
        data = {
            "ub_id": ub_id,
            "user_id": user_id,
            "block_id": block_id,
            "text": text,
            "user_files": user_files,
            "status": "new"
        }
        if created_at is not None:
            data["created_at"] = created_at
        try:
            response = await self.client.post(f"{self.base_url}/add_air", json=data)
            if response.status_code in [200, 201]:
                return response.json()
        except Exception as e:
            print(f"Error adding air message: {e}")
        return {}

    async def update_air_message(self, air_id: int, ai_content: List[Dict], status: str = "completed") -> Dict[str, Any]:
        data = {
            "id": air_id,
            "ai_content": ai_content,
            "status": status
        }
        try:
            response = await self.client.patch(f"{self.base_url}/update_air", json=data)
            if response.status_code in [200, 201]:
                return response.json()
        except Exception as e:
            print(f"Error updating air message: {e}")
        return {}

    async def get_all_workflow_states_with_answers(self) -> List[Dict[str, Any]]:
        """Returns all workflow_state records that still have non-empty answers (pre-migration data)."""
        try:
            response = await self.client.get(f"{self.base_url}/workflow_state")
            if response.status_code == 200:
                data = response.json()
                items = data if isinstance(data, list) else data.get("items", data.get("data", []))
                result = []
                for item in items:
                    raw = item.get("answers", "[]")
                    try:
                        answers = json.loads(raw) if isinstance(raw, str) else (raw or [])
                    except Exception:
                        answers = []
                    if answers:
                        item["_parsed_answers"] = answers
                        result.append(item)
                return result
        except Exception as e:
            print(f"Error getting all workflow states: {e}")
        return []

    async def get_messages(self, ub_id: int) -> List[Dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}/air", params={"ub_id": ub_id})
        response.raise_for_status()
        return response.json()
    
    async def save_message_pair(self, ub_id: int, user_message: str, ai_response: str, prev_id: Optional[int] = None) -> Dict[str, Any]:
        timestamp = int(datetime.now().timestamp() * 1000)
        message_record = {
            "ub_id": ub_id,
            "created_at": timestamp,
            "status": "new",
            "user_content": json.dumps({"type": "text", "text": user_message, "created_at": timestamp}),
            "ai_content": json.dumps([{"text": ai_response, "title": "", "created_at": timestamp}]),
            "prev_id": prev_id if prev_id else 0
        }
        response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
        return response.json() if response.status_code in [200, 201] else {"id": timestamp}
    
    def _extract_score(self, evaluation_text: str) -> Optional[float]:
        patterns = [
            r'\*\*Total Score:\*\*\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Total Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Загальна оцінка:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text)
            if match:
                try:
                    score = float(match.group(1))
                    max_score = float(match.group(2))
                    if max_score > 0:
                        return round(score, 2)
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    def _parse_grading_output(self, evaluation_text: str) -> List[Dict[str, Any]]:
        grading_output = []
        
        criterion_pattern = r'##\s*Criterion\s*\d+[:\s]*([^\n]+)\n(.*?)(?=##\s*Criterion|\#\s*Summary|$)'
        matches = re.findall(criterion_pattern, evaluation_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            criterion_name = match[0].strip()
            criterion_block = match[1]
            
            grade_pattern = r'\*\*Grade:\*\*\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*points?'
            grade_match = re.search(grade_pattern, criterion_block, re.IGNORECASE)
            
            grade = 0
            max_points = 0
            if grade_match:
                try:
                    grade = float(grade_match.group(1))
                    max_points = float(grade_match.group(2))
                except ValueError:
                    pass
            
            assessment_pattern = r'\*\*Assessment:\*\*\s*([^\*]+?)(?=\*\*|$)'
            assessment_match = re.search(assessment_pattern, criterion_block, re.DOTALL | re.IGNORECASE)
            assessment = assessment_match.group(1).strip() if assessment_match else ""
            
            reasoning_pattern = r'\*\*Reasoning:\*\*\s*([^\*]+?)(?=\*\*|$)'
            reasoning_match = re.search(reasoning_pattern, criterion_block, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            grading_output.append({
                "criterion_name": criterion_name,
                "grade": int(grade) if grade == int(grade) else grade,
                "max_points": int(max_points) if max_points == int(max_points) else max_points,
                "summary": assessment[:500] if assessment else "",
                "grading_comment": reasoning[:500] if reasoning else ""
            })
        
        return grading_output
    
    async def _parse_grading_with_ai(self, evaluation_text: str) -> List[Dict[str, Any]]:
        if not self.openai_api_key:
            print("WARNING: No OpenAI API key available for AI parsing")
            return []
        
        try:
            client = OpenAI(api_key=self.openai_api_key)
            
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract all criterion grades from the evaluation text. For each criterion, identify the name, grade received, maximum points possible, assessment summary, and grading comment/reasoning."
                    },
                    {
                        "role": "user",
                        "content": f"Parse this evaluation and extract all criteria with their grades:\n\n{evaluation_text}"
                    }
                ],
                response_format=EvaluationParsed,
            )
            
            parsed = completion.choices[0].message.parsed
            
            if not parsed or not parsed.criteria:
                print("WARNING: AI parsing returned no criteria")
                return []
            
            grading_output = []
            for criterion in parsed.criteria:
                grading_output.append({
                    "criterion_name": criterion.criterion_name,
                    "grade": int(criterion.grade) if criterion.grade == int(criterion.grade) else criterion.grade,
                    "max_points": int(criterion.max_points) if criterion.max_points == int(criterion.max_points) else criterion.max_points,
                    "summary": criterion.summary[:500] if criterion.summary else "",
                    "grading_comment": criterion.grading_comment[:500] if criterion.grading_comment else ""
                })
            
            print(f"AI successfully parsed {len(grading_output)} criteria")
            return grading_output
            
        except Exception as e:
            print(f"ERROR: AI parsing failed: {type(e).__name__}: {str(e)}")
            return []
    
    async def update_chat_status(
        self, 
        ub_id: int, 
        status: Optional[ChatStatus] = None, 
        grade: Optional[str] = None, 
        last_air_id: Optional[int] = None
    ):
        update_data = {"ub_id": int(ub_id)}
        
        if status:
            update_data["status"] = status.value
            update_data["set_status"] = True
        
        if grade is not None:
            update_data["work_summary"] = grade
            
            score = self._extract_score(grade)
            if score is not None:
                update_data["grade"] = score
                print(f"Extracted score: {score}")
            else:
                print("Could not extract numerical score from evaluation")
            
            grading_output = self._parse_grading_output(grade)

            if not grading_output or all(c.get('grade', 0) == 0 and c.get('max_points', 0) == 0 for c in grading_output):
                print("Regex parsing failed or returned invalid grades, trying AI parsing as fallback")
                grading_output = await self._parse_grading_with_ai(grade)
            
            if grading_output:
                update_data["grading_output"] = grading_output
                print(f"Parsed {len(grading_output)} criteria into grading_output")
            else:
                print("Could not parse grading_output from evaluation")
        
        if last_air_id:
            update_data["last_air_id"] = int(last_air_id)
        
        try:
            print(f"Updating UB {ub_id} with data keys: {list(update_data.keys())}")
            response = await self.client.post(f"{self.base_url}/update_ub", json=update_data)
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"Update UB successful: {result}")
                return result
            else:
                print(f"Update UB error: {response.status_code}")
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"Status update error: {e}")
            import traceback
            traceback.print_exc()
        return None

    async def save_token_usage(
        self,
        ub_id: int,
        block_id: int,
        course_id: int,
        user_id: int,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o",
        operation_type: str = "chat"
    ) -> Optional[Dict[str, Any]]:
        data = {
            "ub_id": ub_id,
            "block_id": block_id,
            "course_id": course_id,
            "user_id": user_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "operation_type": operation_type
        }
        try:
            response = await self.client.post(f"{self.base_url}/token_usage", json=data)
            if response.status_code in [200, 201]:
                return response.json()
            else:
                print(f"Save token usage error: {response.status_code}")
        except Exception as e:
            print(f"Token usage save error: {e}")
        return None

    async def get_course_token_usage(self, course_id: int) -> Dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/token_usage/course/{course_id}")
            if response.status_code == 200:
                records = response.json()
                total_input = sum(r.get("input_tokens", 0) for r in records)
                total_output = sum(r.get("output_tokens", 0) for r in records)
                total_tokens = sum(r.get("total_tokens", 0) for r in records)
                total_requests = len(records)
                cost_input = (total_input / 1000000) * 2.50
                cost_output = (total_output / 1000000) * 10.00
                estimated_cost_usd = round(cost_input + cost_output, 4)
                return {
                    "course_id": course_id,
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_tokens": total_tokens,
                    "total_requests": total_requests,
                    "estimated_cost_usd": estimated_cost_usd
                }
        except Exception as e:
            print(f"Get course token usage error: {e}")
        return {
            "course_id": course_id,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_requests": 0,
            "estimated_cost_usd": 0
        }

    async def get_course_token_usage_by_block(self, course_id: int) -> Dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/token_usage/course/{course_id}/by_block")
            if response.status_code == 200:
                records = response.json()
                by_block = {}
                for r in records:
                    bid = r.get("block_id")
                    if bid not in by_block:
                        by_block[bid] = {"block_id": bid, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}
                    by_block[bid]["input_tokens"] += r.get("input_tokens", 0)
                    by_block[bid]["output_tokens"] += r.get("output_tokens", 0)
                    by_block[bid]["total_tokens"] += r.get("total_tokens", 0)
                    by_block[bid]["requests"] += 1
                total_tokens = sum(b["total_tokens"] for b in by_block.values())
                return {
                    "course_id": course_id,
                    "total_tokens": total_tokens,
                    "by_block": list(by_block.values())
                }
        except Exception as e:
            print(f"Get course token usage by block error: {e}")
        return {"course_id": course_id, "total_tokens": 0, "by_block": []}

    async def get_user_token_usage(self, course_id: int, user_id: int) -> Dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/token_usage/course/{course_id}/user/{user_id}")
            if response.status_code == 200:
                records = response.json()
                total_input = sum(r.get("input_tokens", 0) for r in records)
                total_output = sum(r.get("output_tokens", 0) for r in records)
                total_tokens = sum(r.get("total_tokens", 0) for r in records)
                total_requests = len(records)
                return {
                    "course_id": course_id,
                    "user_id": user_id,
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_tokens": total_tokens,
                    "total_requests": total_requests
                }
        except Exception as e:
            print(f"Get user token usage error: {e}")
        return {
            "course_id": course_id,
            "user_id": user_id,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_requests": 0
        }

    async def get_course_token_usage_by_period(
        self,
        course_id: int,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        try:
            response = await self.client.get(
                f"{self.base_url}/token_usage/course/{course_id}/period",
                params={"start_date": start_date, "end_date": end_date}
            )
            if response.status_code == 200:
                records = response.json()
                total_input = sum(r.get("input_tokens", 0) for r in records)
                total_output = sum(r.get("output_tokens", 0) for r in records)
                total_tokens = sum(r.get("total_tokens", 0) for r in records)
                total_requests = len(records)
                cost_input = (total_input / 1000000) * 2.50
                cost_output = (total_output / 1000000) * 10.00
                estimated_cost_usd = round(cost_input + cost_output, 4)
                return {
                    "course_id": course_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_tokens": total_tokens,
                    "total_requests": total_requests,
                    "estimated_cost_usd": estimated_cost_usd
                }
        except Exception as e:
            print(f"Get course token usage by period error: {e}")
        return {
            "course_id": course_id,
            "start_date": start_date,
            "end_date": end_date,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_requests": 0,
            "estimated_cost_usd": 0
        }