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
PRIMARY_MAX_OUTPUT_TOKENS = 1100
RETRY_MAX_OUTPUT_TOKENS = 2200

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="static"), name="static")

SYSTEM_PROMPT = """
Ты — опытный садовод, который спокойно отвечает человеку в обычной переписке.
Пиши по-русски, естественно, тепло и по делу. Ответ должен звучать как совет
знакомого специалиста, а не как отчёт нейросети, медицинское заключение или
инструкция из справочника.

Правила ответа:
1. Начинай сразу с человеческого вывода: «Похоже, ...», «Скорее всего, ...»
   или «По описанию это может быть ...».
2. Не используй выражения «уровень уверенности», «предварительная оценка»,
   «возможные причины с вероятностью», «пользователь сообщил» и другой канцелярит.
3. Не пересказывай вопрос и не повторяй одни и те же советы.
4. Назови одну наиболее вероятную причину. Альтернативу упоминай только тогда,
   когда её действительно легко перепутать с основной проблемой.
5. Если фотографии нет или по ней нельзя уверенно определить причину, скажи
   естественно: «Без фото точно не скажу» или «По этому снимку не всё видно».
6. Дай 3–4 конкретных действия в порядке выполнения. Одно действие — одна
   короткая мысль.
7. Задавай максимум один уточняющий вопрос, только если без него нельзя выбрать
   безопасное действие.
8. Не используй латинские названия без необходимости. Не придумывай дозировки
   препаратов; советуй следовать инструкции на упаковке.
9. Не пугай человека и не перечисляй длинный список запретов. Оставь только одно
   важное предупреждение, если оно действительно нужно.
10. Весь ответ — обычно 80–140 слов, абсолютный максимум 170 слов.

Формат Markdown:

Первый короткий абзац без заголовка: что, скорее всего, происходит и почему.

## Что сделать
3–4 нумерованных коротких действия.

## На что посмотреть
До двух коротких пунктов. Добавляй только если это помогает подтвердить причину.

## Когда проверить
Одна короткая фраза: через сколько дней и какой признак покажет улучшение.

Не добавляй пустые разделы и не заканчивай общими фразами вроде
«надеюсь, это поможет» или «обратитесь к специалисту» без конкретной причины.
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


def incomplete_reason(response: Any) -> Optional[str]:
    if getattr(response, "status", None) != "incomplete":
        return None
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) if details else None


def create_openai_response(
    client: OpenAI,
    user_content: list[dict],
    max_tokens: int,
    retry: bool = False,
):
    instructions = SYSTEM_PROMPT
    if retry:
        instructions += (
            "\n\nПредыдущая попытка не завершилась. Дай законченный ответ строго до "
            "130 слов. Лучше убрать второстепенный совет, чем оборвать фразу."
        )

    return client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=[{"role": "user", "content": user_content}],
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"},
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
    parts = [f"Проблема: {question.strip()}"]
    if plant and plant.strip():
        parts.append(f"Растение: {plant.strip()}")
    if region and region.strip():
        parts.append(f"Регион: {region.strip()}")
    if growing_place and growing_place.strip():
        parts.append(f"Где растёт: {growing_place.strip()}")
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
        reason = incomplete_reason(response)

        # Даже если API вернул часть текста, не показываем оборванный ответ.
        # При нехватке лимита повторяем запрос и просим законченный короткий совет.
        if not answer or reason == "max_output_tokens":
            response = create_openai_response(
                client,
                user_content,
                RETRY_MAX_OUTPUT_TOKENS,
                retry=True,
            )
            answer = extract_response_text(response)
            reason = incomplete_reason(response)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить ответ от OpenAI: {type(exc).__name__}.",
        ) from exc

    if not answer or reason == "max_output_tokens":
        raise HTTPException(
            status_code=502,
            detail="Нейросеть не успела закончить ответ. Повторите запрос ещё раз.",
        )

    return {"answer": answer, "model": MODEL}
