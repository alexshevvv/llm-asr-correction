# LLM-ASR-Correction

[![Status](https://img.shields.io/badge/status-in--development-yellow.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Распознавание речи с корректировкой ошибок с использованием больших языковых моделей

*Дипломная работа — ВШЭ Нижний Новгород, 2026*

---

## Описание

Исследование эффективности больших языковых моделей (LLM) для коррекции ошибок автоматического распознавания речи (ASR). Проект сравнивает несколько ASR-моделей на русских и английских датасетах, применяет LLM-коррекцию и анализирует улучшение метрик качества.

## Организация проекта

Проект построен как научно-исследовательская работа с двумя параллельными слоями:

- **`src/` + `scripts/`** — модульный код: ASR-обёртки, LLM-клиенты, метрики, утилиты. Переиспользуемые компоненты, покрытые тестами.
- **`notebooks/`** — Jupyter-ноутбуки для каждого этапа исследования. Ноутбуки импортируют модули из `src/` и сохраняют результаты в `experiments/`.

По мере развития работы добавляются новые ноутбуки, расширяются модули в `src/`, а результаты накапливаются в `experiments/results/`.

## Результаты baseline

| Модель | Датасет | Язык | Mean WER | Mean CER |
|--------|---------|------|----------|----------|
| Whisper-base | LibriSpeech test-clean | EN | 5.69% | 2.41% |
| Whisper-base | FLEURS | RU | 22.26% | 6.25% |
| GigaAM-v2-CTC | FLEURS | RU | 9.84% | 4.38% |

## Результаты LLM-коррекции (Llama 3.1 8B)

| Эксперимент | Baseline WER | + LLM | Изменение |
|-------------|-------------|-------|-----------|
| Whisper EN | 11.39% | 16.14% | -41.8% |
| Whisper RU | 23.68% | 21.22% | +10.4% |
| GigaAM RU | 16.97% | 19.82% | -16.8% |

## Структура проекта
```text
llm-asr-correction/
├── src/                        # Исходный код (модули)
│   ├── asr/                    # ASR модели (Whisper, GigaAM)
│   ├── correction/             # LLM коррекция ошибок
│   ├── evaluation/             # Метрики (WER, CER)
│   └── utils/                  # Конфигурация, аудио, датасеты
├── scripts/                    # Скрипты запуска экспериментов
├── notebooks/                  # Jupyter-ноутбуки исследования
│   └── 01_baseline_asr.ipynb   # Демо: baseline + LLM-коррекция
├── experiments/                # Результаты экспериментов
│   ├── configs/                # Конфигурации запусков
│   ├── results/                # CSV с метриками
│   └── logs/                   # Логи экспериментов
├── tests/                      # Unit-тесты
├── data/                       # Датасеты (не коммитятся)
│   ├── raw/                    # Исходные аудиофайлы
│   ├── processed/              # Обработанные данные
│   └── samples/                # Примеры для тестов
└── docs/                       # Документация
```

## Установка

### Системные требования

- Python 3.10+
- CUDA-совместимый GPU (рекомендуется)
- ffmpeg

### Настройка окружения
```bash
git clone https://github.com/alexshevvv/llm-asr-correction.git
cd llm-asr-correction
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp env.example .env
# Заполнить .env своими API-ключами
```

GigaAM устанавливается отдельно:
```bash
pip install git+https://github.com/salute-developers/GigaAM.git
```

### Получение API-ключей

- **Groq** (бесплатно): [console.groq.com/keys](https://console.groq.com/keys)

## Запуск тестов
```bash
pytest tests/ -v
```

## Дорожная карта

- [x] Baseline ASR: Whisper + GigaAM на EN/RU датасетах
- [x] LLM-коррекция: Llama 3.1 8B через Groq API
- [ ] Бенчмарк LLM: GPT-4o-mini, Claude, Gemini, Llama 70B
- [ ] Confidence-aware коррекция
- [ ] Оптимизация промптов (few-shot, domain-specific)
- [ ] Масштабирование до 500–1000 сэмплов
- [ ] Анализ по типам ошибок

## Автор

**Алексей Шевченко** — ВШЭ Нижний Новгород

## Лицензия

MIT License