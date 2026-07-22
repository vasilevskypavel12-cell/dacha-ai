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
  answerStatus.textContent = 'Смотрю описание и фотографию…';
}

function resetLoading() {
  loadingState.hidden = true;
  submitButton.disabled = false;
  submitButton.querySelector('span').textContent = 'Получить рекомендацию';
}


function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatInline(value) {
  return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function renderAnswer(markdown) {
  const lines = markdown.replace(/\r/g, '').split('\n');
  const sections = [];
  let section = { title: '', items: [], paragraphs: [] };

  const pushSection = () => {
    if (section.title || section.items.length || section.paragraphs.length) {
      sections.push(section);
    }
    section = { title: '', items: [], paragraphs: [] };
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith('## ')) {
      pushSection();
      section.title = line.slice(3).trim();
      continue;
    }

    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    const bullet = line.match(/^[-•]\s+(.+)$/);
    if (numbered) {
      section.items.push({ type: 'number', text: numbered[1] });
    } else if (bullet) {
      section.items.push({ type: 'bullet', text: bullet[1] });
    } else {
      section.paragraphs.push(line);
    }
  }
  pushSection();

  if (!sections.length) {
    answerElement.textContent = markdown;
    return;
  }

  answerElement.innerHTML = sections.map((item, index) => {
    const title = item.title
      ? `<h3>${formatInline(item.title)}</h3>`
      : index === 0 ? '<h3>Рекомендация</h3>' : '';
    const paragraphs = item.paragraphs
      .map((text) => `<p>${formatInline(text)}</p>`)
      .join('');

    const numberedItems = item.items.filter((entry) => entry.type === 'number');
    const bulletItems = item.items.filter((entry) => entry.type === 'bullet');
    const numberedList = numberedItems.length
      ? `<ol>${numberedItems.map((entry) => `<li>${formatInline(entry.text)}</li>`).join('')}</ol>`
      : '';
    const bulletList = bulletItems.length
      ? `<ul>${bulletItems.map((entry) => `<li>${formatInline(entry.text)}</li>`).join('')}</ul>`
      : '';

    const normalizedTitle = item.title.toLowerCase();
    const importantClass = normalizedTitle.includes('важно') ? ' answer-card--important' : '';
    const leadClass = !item.title && index === 0 ? ' answer-card--lead' : '';
    return `<section class="answer-card${importantClass}${leadClass}">${title}${paragraphs}${numberedList}${bulletList}</section>`;
  }).join('');
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

    renderAnswer(payload.answer);
    answerElement.hidden = false;
    answerStatus.textContent = 'Готово. Ниже — самое важное без лишней теории.';
  } catch (error) {
    errorElement.textContent = error.message || 'Произошла неизвестная ошибка.';
    errorElement.hidden = false;
    answerStatus.textContent = 'Не удалось выполнить запрос.';
  } finally {
    resetLoading();
  }
});
