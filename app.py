import base64
import os
from typing import Optional

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

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="static"), name="static")

SYSTEM_PROMPT = """
Ты — осторожный и практичный помощник для садоводов и огородников.
Отвечай по-русски, простыми словами и без лишней воды.

Правила ответа:
1. Не утверждай диагноз со 100% уверенностью только по фотографии.
2. Сначала назови наиболее вероятные причины и уровень уверенности.
3. Если данных недостаточно, перечисли 2–4 уточняющих вопроса.
4. Затем дай безопасный пошаговый план: что проверить сегодня, что сделать,
   чего не делать и когда оценить результат повторно.
5. Не придумывай точные дозировки средств защиты растений. При упоминании
   препарата проси следовать официальной инструкции на упаковке и учитывать
   срок ожидания до сбора урожая.
6. Учитывай культуру, регион, место выращивания и описание пользователя.
7. Если на фото не растение или фото недостаточно качественное, прямо скажи об этом.
8. Напомни, что ответ является предварительной рекомендацией, а при массовом
   поражении или риске потери урожая лучше обратиться к агроному.

Структура:
- Краткий вывод
- Возможные причины
- Что проверить
- Что сделать сейчас
- Чего не делать
- Когда проверить результат
""".strip()


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
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": user_content}],
            max_output_tokens=1400,
        )
    except Exception as exc:
        # Не отправляем пользователю содержимое ключа или внутренние данные.
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить ответ от OpenAI: {type(exc).__name__}.",
        ) from exc

    answer = (response.output_text or "").strip()
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Нейросеть вернула пустой ответ. Попробуйте ещё раз.",
        )

    return {"answer": answer, "model": MODEL}
