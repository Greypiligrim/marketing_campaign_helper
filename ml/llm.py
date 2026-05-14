import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import dateparser

from config import settings
from ml import prompts

logger = logging.getLogger(__name__)


def _extract_json(content: str) -> str:
    """Достаёт JSON из ответа LLM: снимает markdown-ограждение и обрезает по скобкам."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return s


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

    async def parse_campaign(
        self,
        user_message: str,
        user_display: Optional[str] = None,
        roster: str = "",
    ) -> Dict[str, Any]:
        try:
            prompt_text = prompts.PARSE_CAMPAIGN.format(
                user_message=user_message,
                user_context=user_display or "не представился",
                current_date=datetime.utcnow().date().isoformat(),
                roster=roster or "(пока никто не представился)",
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.0,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            logger.info(f"Parse response: {content}")
            
            data = json.loads(_extract_json(content))
            
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

    async def pick_template(
        self, campaign_name: str, task: Optional[str], templates: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Просит LLM выбрать подходящий шаблон. Возвращает имя шаблона или None."""
        if not templates:
            return None
        try:
            names = {t["name"] for t in templates}
            templates_list = "\n".join(
                f"- {t['name']}: {t.get('description') or ''} "
                f"(подзадач: {len(t.get('subtasks') or [])})"
                for t in templates
            )
            prompt_text = prompts.PICK_TEMPLATE.format(
                campaign_name=campaign_name,
                task=task or "",
                templates_list=templates_list,
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.0,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"].strip().strip('"').strip()
            logger.info(f"Pick template response: {content}")
            if content in names:
                return content
            # На случай, если модель добавила лишний текст — ищем имя как подстроку.
            for name in names:
                if name.lower() in content.lower():
                    return name
            return None
        except Exception as e:
            logger.error(f"pick_template failed: {e}", exc_info=True)
            return None

    async def decompose_task_llm(
        self,
        task_name: str,
        participants: List[str],
        current_date: str,
        deadline: str, 
        days_to_deadline: int,
        budget_rub: float,
        template_examples: str = ""
    ) -> List[Dict[str, Any]]:
        try:
            prompt_text = prompts.DECOMPOSE_TASK.format(
                task_name=task_name,
                participants=", ".join(participants),
                current_date=current_date,
                deadline=deadline,
                days_to_deadline=days_to_deadline,
                budget_rub=budget_rub,
                template_examples=template_examples if template_examples else "(Нет примеров)",
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
            data = json.loads(_extract_json(content))
            subtasks = data.get("subtasks", [])
            
            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            return subtasks
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decompose JSON: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"decompose_task_llm failed: {e}", exc_info=True)
            raise

    async def answer_question(
        self,
        question: str,
        campaigns_context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Отвечает на вопрос с учётом контекста кампаний и истории диалога.

        ``history`` — список реплик ``[{"role": "user"|"assistant", "content": str}, ...]``
        в хронологическом порядке. Последняя реплика пользователя из истории НЕ дублирует
        текущий ``question`` — она уже сохранена выше по стеку.
        """
        try:
            system_text = prompts.QA_SYSTEM.format(campaigns_context=campaigns_context)
            messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
            for m in (history or [])[-12:]:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": question})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            }
            resp = await self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            logger.error(f"answer_question failed: {e}", exc_info=True)
            return "Извините, не удалось получить ответ."