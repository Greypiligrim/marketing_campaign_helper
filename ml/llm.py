import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import dateparser

from config import settings
from ml import prompts

logger = logging.getLogger(__name__)


def _normalize_username(username: str) -> str:
    return username.lstrip("@").strip()


def _iso_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    
    # If it's already in ISO format (YYYY-MM-DD), validate and return as-is
    if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        try:
            datetime.fromisoformat(date_str)
            return date_str
        except ValueError:
            pass
    
    # Otherwise, try parsing with dateparser
    parsed = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
    if not parsed:
        return None
    return parsed.date().isoformat()


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class NemotronLLM:
    def __init__(self) -> None:
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.OPENROUTER_MODEL
        self.base_url = settings.OPENROUTER_URL + "/chat/completions"
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with session.post(self.base_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"API error: {resp.status} {text}")
                    raise Exception(f"API error: {resp.status} {text}")
                data = await resp.json()
                return data

    async def classify_message(self, user_message: str) -> str:
        try:
            prompt_text = prompts.CLASSIFY_MESSAGE.format(user_message=user_message)
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.0,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"].strip().lower()
            logger.info(f"Classify response: {content}")
            
            if "campaign" in content:
                return "campaign"
            elif "question" in content:
                return "question"
            else:
                return "other"
        except Exception as e:
            logger.error(f"classify_message failed: {e}", exc_info=True)
            return "question"

    async def parse_campaign(self, user_message: str) -> Dict[str, Any]:
        try:
            prompt_text = prompts.PARSE_CAMPAIGN.format(user_message=user_message)
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.0,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            logger.info(f"Parse response: {content}")
            
            data = json.loads(content)
            
            return {
                "campaign_name": data.get("campaign_name"),
                "task": data.get("task"),
                "deadline": _iso_date(data.get("deadline")),
                "participants": [_normalize_username(x) for x in (data.get("participants") or []) if x],
                "budget_rub": _float_or_none(data.get("budget_rub")),
                "confidence_score": float(data.get("confidence_score", 0.0)),
            }
        except Exception as e:
            logger.error(f"parse_campaign failed: {e}", exc_info=True)
            raise

    async def decompose_task_llm(
        self, 
        task_name: str, 
        participants: List[str],
        current_date: str,
        deadline: str, 
        days_to_deadline: int,
        budget_rub: float
    ) -> List[Dict[str, Any]]:
        try:
            prompt_text = prompts.DECOMPOSE_TASK.format(
                task_name=task_name,
                participants=", ".join(participants),
                current_date=current_date,
                deadline=deadline,
                days_to_deadline=days_to_deadline,
                budget_rub=budget_rub,
            )
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.0,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            logger.info(f"Decompose response: {content}")

            # Попытка спарсить JSON из ответа
            data = json.loads(content)
            subtasks = data.get("subtasks", [])
            
            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            return subtasks
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decompose JSON: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"decompose_task_llm failed: {e}", exc_info=True)
            raise

    async def answer_question(self, question: str, campaigns_context: str) -> str:
        try:
            prompt_text = prompts.QA_PROMPT.format(
                campaigns_context=campaigns_context,
                question=question,
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.2,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            logger.error(f"answer_question failed: {e}", exc_info=True)
            return "Извините, не удалось получить ответ."