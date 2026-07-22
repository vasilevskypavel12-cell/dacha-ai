import base64
import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv()

APP_TITLE = "Дачный советник"
MAX_IMAGE_SIZE = 8 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
PRIMARY_MAX_OUTPUT_TOKENS = 1400
RETRY_MAX_OUTPUT_TOKENS = 2400

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="static"), name="static")

SYSTEM_PROMPT = """
Ты — практичный помощник для садоводов. Отвечай по-русски, коротко,
понятно и без канцелярита. Пользователь должен за 20 секунд понять,
что вероятнее всего произошло и что делать дальше.

Основные правила:
1. Не повторяй сведения пользователя и не пересказывай вопрос.
2. Не используй латинские названия, если без них можно обойтись.
3. Не перечисляй много маловероятных причин. Назови одну основную причину
   и максимум две альтернативы только тогда, когда они действительно важны.
4. Не дублируй один совет в разных разделах.
5. Не пиши длинных вступлений и общих фраз.
6. Если фотографии нет или её недостаточно, одной короткой фразой укажи,
   что оценка предварительная, и задай максимум два самых полезных вопроса.
7. Не придумывай дозировки препаратов. При необходимости рекомендуй средство
   по типу действия и проси соблюдать инструкцию производителя.
8. Давай только безопасные действия. При серьёзном или массовом поражении
   коротко укажи, когда нужен агроном.
9. Обычный ответ должен занимать 120–220 слов. Не добавляй разделы,
   в которых нет полезной информации.

Всегда используй этот формат Markdown:

## Вероятная проблема
Один короткий вывод. Укажи уверенность словами: высокая, средняя или низкая.
Если фото нет, добавь: «Без фотографии оценка предварительная».

## Что сделать сейчас
От 3 до 5 конкретных нумерованных действий в правильном порядке.

## Что проверить
До 3 коротких пунктов. Добавляй этот раздел только при необходимости.

## Когда оценить результат
Один короткий срок и понятный признак улучшения.

## Важно
Одна короткая мера предосторожности. Не повторяй стандартные предупреждения.
""".strip()


def extract_response_text(response: Any) -> str:
    """Извлекает итоговый текст из Responses API с резервным разбором output."""
    direct_text = getattr(response, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()


def create_openai_response(client: OpenAI, user_content: list[dict], max_tokens: int):
    return client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": user_content}],
        reasoning={"effort": "minimal"},
        max_output_tokens=max_tokens,
    )


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="На сервере не задан OPENAI_API_KEY. Добавьте ключ в файл .env.",
        )
    return OpenAI(api_key=api_key)


def build_user_text(
    question: str,
    plant: Optional[str],
    region: Optional[str],
    growing_place: Optional[str],
) -> str:
    parts = [f"Проблема пользователя: {question.strip()}"]
    if plant and plant.strip():
        parts.append(f"Растение или культура: {plant.strip()}")
    if region and region.strip():
        parts.append(f"Регион: {region.strip()}")
    if growing_place and growing_place.strip():
        parts.append(f"Место выращивания: {growing_place.strip()}")
    return "\n".join(parts)


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/api/ask")
async def ask_ai(
    question: str = Form(...),
    plant: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    growing_place: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> dict:
    question = question.strip()
    if len(question) < 5:
        raise HTTPException(status_code=400, detail="Опишите проблему подробнее.")
    if len(question) > 4000:
        raise HTTPException(status_code=400, detail="Описание слишком длинное.")

    user_content = [
        {
            "type": "input_text",
            "text": build_user_text(question, plant, region, growing_place),
        }
    ]

    if image and image.filename:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Поддерживаются только JPG, PNG и WEBP.",
            )

        image_bytes = await image.read(MAX_IMAGE_SIZE + 1)
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Фотография больше 8 МБ. Уменьшите её размер.",
            )
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл фотографии пустой.")

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:{image.content_type};base64,{encoded}",
                "detail": "high",
            }
        )

    client = get_client()

    try:
        response = create_openai_response(
            client, user_content, PRIMARY_MAX_OUTPUT_TOKENS
        )
        answer = extract_response_text(response)

        # GPT-5 mini иногда расходует малый лимит на рассуждение и не успевает
        # вывести текст. В таком случае автоматически повторяем запрос с запасом.
        if not answer:
            response = create_openai_response(
                client, user_content, RETRY_MAX_OUTPUT_TOKENS
            )
            answer = extract_response_text(response)
    except Exception as exc:
        # Не отправляем пользователю содержимое ключа или внутренние данные.
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить ответ от OpenAI: {type(exc).__name__}.",
        ) from exc

    if not answer:
        status = getattr(response, "status", "неизвестен")
        incomplete = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None) if incomplete else None
        safe_reason = f" Причина: {reason}." if reason else ""
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenAI не сформировал текст ответа после повторной попытки. "
                f"Статус: {status}.{safe_reason}"
            ),
        )

    return {"answer": answer, "model": MODEL}
