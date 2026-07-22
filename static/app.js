const form = document.getElementById('diagnosis-form');
const imageInput = document.getElementById('image');
const preview = document.getElementById('preview');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const removeImageButton = document.getElementById('remove-image');
const submitButton = document.getElementById('submit-button');
const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const answerElement = document.getElementById('answer');
const errorElement = document.getElementById('error');
const answerStatus = document.getElementById('answer-status');

let previewUrl = null;

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imageInput.value = '';
  preview.removeAttribute('src');
  preview.hidden = true;
  removeImageButton.hidden = true;
  uploadPlaceholder.hidden = false;
}

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return clearPreview();

  if (file.size > 8 * 1024 * 1024) {
    alert('Фотография больше 8 МБ. Выберите файл меньшего размера.');
    return clearPreview();
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  removeImageButton.hidden = false;
  uploadPlaceholder.hidden = true;
});

removeImageButton.addEventListener('click', (event) => {
  event.preventDefault();
  event.stopPropagation();
  clearPreview();
});

function showLoading() {
  emptyState.hidden = true;
  answerElement.hidden = true;
  errorElement.hidden = true;
  loadingState.hidden = false;
  submitButton.disabled = true;
  submitButton.querySelector('span').textContent = 'Анализируем…';
  answerStatus.textContent = 'ИИ изучает описание и фотографию.';
}

function resetLoading() {
  loadingState.hidden = true;
  submitButton.disabled = false;
  submitButton.querySelector('span').textContent = 'Получить рекомендацию';
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  showLoading();

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      body: new FormData(form),
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error('Сервер вернул некорректный ответ.');
    }

    if (!response.ok) {
      throw new Error(payload.detail || 'Не удалось получить рекомендацию.');
    }

    answerElement.textContent = payload.answer;
    answerElement.hidden = false;
    answerStatus.textContent = `Ответ сформирован моделью ${payload.model}.`;
  } catch (error) {
    errorElement.textContent = error.message || 'Произошла неизвестная ошибка.';
    errorElement.hidden = false;
    answerStatus.textContent = 'Не удалось выполнить запрос.';
  } finally {
    resetLoading();
  }
});
