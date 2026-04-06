# LLM-ASR-Correction

[![Status](https://img.shields.io/badge/status-in--development-yellow.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/docs/inference-providers)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991.svg?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![seaborn](https://img.shields.io/badge/seaborn-0.13+-4C72B0.svg)](https://seaborn.pydata.org/)

Распознавание речи с корректировкой ошибок с использованием больших языковых моделей

*Дипломная работа --- ВШЭ Нижний Новгород, 2026*

---

## Описание

Исследование эффективности больших языковых моделей (LLM) для коррекции ошибок
автоматического распознавания речи (ASR). Проект сравнивает несколько ASR-моделей
на русских и английских датасетах, применяет LLM-коррекцию через HuggingFace
Inference API и анализирует улучшение метрик качества (WER, CER).

## Организация проекта

Проект построен как научно-исследовательская работа с двумя параллельными слоями:

- **Локальный проект (PyCharm)** - модульный код в `src/` и `scripts/`,
  покрытый тестами. ASR-модели Whisper и Wav2Vec2 работают на CPU.
  LLM-коррекция через HuggingFace Inference API.
  Результаты сохраняются в `experiments/results/`.

- **Google Colab** - ноутбуки в `notebooks/` с расширенным набором ASR-моделей
  (включая GigaAM, требующий CUDA). Ноутбуки используют те же LLM через
  HuggingFace Inference API и формируют дополнительные результаты.

Оба слоя дополняют друг друга: локальный проект обеспечивает воспроизводимость
и тестируемость, Colab - доступ к GPU-моделям и быстрое прототипирование.

## Структура проекта

```text
llm-asr-correction/
├── data/                           # Данные (не коммитятся)
│   ├── raw/                        # Кеш HuggingFace datasets
│   ├── processed/                  # Кешированные baseline-результаты
│   └── samples/                    # Аудио-примеры (EN/RU)
│ 
├── experiments/                    # Результаты экспериментов
│   ├── configs/                    # JSON-конфиги каждого запуска
│   ├── results/                    # CSV-результаты + PNG-графики
│   └── logs/                       # Логи запусков
│
├── notebooks/                      # Jupyter-ноутбуки (Colab)
│   ├── 01_baseline_asr.ipynb       # Первый прототип
│   └── 02_llm_benchmark_demo.ipynb # Полный бенчмарк (5 LLM x 4 ASR)
│
├── scripts/                        # Скрипты запуска экспериментов
│   ├── run_benchmark.py            # Точка входа: полный пайплайн
│   ├── benchmark_baselines.py      # ASR baseline с кешированием
│   ├── benchmark_config.py         # Константы и список LLM
│   ├── benchmark_io.py             # Сохранение CSV, JSON-конфигов
│   ├── benchmark_pipeline.py       # Цикл LLM-коррекции
│   ├── benchmark_summary.py        # Построение сводной таблицы
│   ├── baseline_cache.py           # Кеширование baseline в data/processed
│   ├── run_baseline.py             # Запуск ASR на датасете
│   └── run_correction.py           # Применение LLM-коррекции
│
├── src/                            # Исходный код (модули)
│   ├── asr/                        # ASR-модели
│   │   ├── base.py                 # Абстрактный интерфейс ASR
│   │   ├── whisper_transcribe.py   # OpenAI Whisper
│   │   ├── wav2vec2_transcribe.py  # Wav2Vec2 XLS-R 1B Russian
│   │   └── gigaam_transcribe.py    # Sber GigaAM (только Colab)
│   ├── correction/                 # LLM-коррекция
│   │   ├── llm_client.py           # HuggingFace Inference API клиент
│   │   ├── clean_response.py       # Постобработка ответов LLM
│   │   └── prompts.py              # Системные и пользовательские промпты
│   ├── evaluation/                 # Метрики
│   │   ├── metrics.py              # WER, CER
│   │   └── normalize.py            # Нормализация текста
│   ├── visualization/              # Визуализация (7 типов графиков)
│   │   ├── common.py               # Общие утилиты и build_viz_df
│   │   ├── wer_change.py           # Grouped barplot: WER Change
│   │   ├── corrected_wer.py        # Faceted: Baseline vs Corrected
│   │   ├── diverging.py            # Horizontal diverging bar
│   │   ├── heatmap.py              # Seaborn heatmap (RdYlGn)
│   │   ├── scatter.py              # Scatter: Baseline WER vs Effect
│   │   ├── baseline.py             # ASR baseline comparison
│   │   ├── radar.py                # LLM performance radar
│   │   └── runner.py               # Запуск всех графиков
│   └── utils/                      # Утилиты
│       ├── config.py               # Конфигурация эксперимента
│       ├── datasets.py             # Загрузка LibriSpeech, FLEURS
│       ├── audio.py                # WAV I/O, ресемплинг
│       └── save_samples.py         # Сохранение аудио-примеров
│
├── tests/                          # Unit-тесты (42 теста)
│   ├── test_audio.py               # Сохранение WAV, непустой файл
│   ├── test_baseline_cache.py      # Кеширование baseline-результатов
│   ├── test_benchmark_io.py        # Создание директорий, запись CSV
│   ├── test_benchmark_summary.py   # Сводная таблица, пустые данные, деградация
│   ├── test_clean_response.py      # Постобработка ответов LLM
│   ├── test_config.py              # Значения по умолчанию, устройство, переопределение
│   ├── test_llm_client.py          # Клиент HF API, fallback при ошибках
│   ├── test_metrics.py             # WER, CER: точное совпадение, ошибки, пустые строки
│   ├── test_normalize.py           # Нормализация текста: регистр, пунктуация, пробелы
│   ├── test_prompts.py             # Промпты EN/RU, язык по умолчанию
│   ├── test_save_samples.py        # Сохранение аудио-примеров, лимит, пропуск существующих
│   └── test_viz_common.py          # Построение DataFrame для визуализации
│
├── .env                            # API-ключи (не коммитится)
├── env.example                     # Шаблон .env
├── requirements.txt                # Зависимости
├── pyproject.toml                  # Конфигурация проекта
└── README.md
```

## ASR-модели

| Модель | Параметры | Тип | Языки | Локально | Colab |
|--------|-----------|-----|-------|----------|-------|
| Whisper base | 74M | Encoder-Decoder | EN, RU | да | да |
| Wav2Vec2 XLS-R 1B | 1B | CTC (fine-tuned) | RU | да | да |
| GigaAM v2 CTC | ~100M | CTC (Sber) | RU | нет | да |

## LLM-модели (HuggingFace Inference API)

| Модель | Параметры | Семейство |
|--------|-----------|-----------|
| Qwen2.5 7B Instruct | 7B | Alibaba |
| Qwen2.5 72B Instruct | 72B | Alibaba |
| GPT-OSS 120B | 120B MoE (5.1B active) | OpenAI |
| Qwen3 235B | 235B MoE (22B active) | Alibaba |
| DeepSeek V3 | 685B MoE | DeepSeek |

## Промежуточные результаты

Бенчмарк проведен в двух средах на 50 сэмплах из LibriSpeech (EN)
и Google FLEURS (RU).

### Локальный запуск (PyCharm, Mac CPU)

Baseline:

| ASR | Датасет | Язык | Mean WER |
|-----|---------|------|----------|
| Whisper base | LibriSpeech test-clean | EN | 6.19% |
| Whisper base | FLEURS | RU | 22.20% |
| Wav2Vec2 XLS-R 1B | FLEURS | RU | 17.80% |

LLM-коррекция (3 ASR x 5 LLM = 15 экспериментов):

| LLM | Whisper EN | Whisper RU | W2V2 RU |
|-----|-----------|-----------|---------|
| Qwen2.5 7B | -19.0% | +3.6% | +12.5% |
| Qwen2.5 72B | -12.4% | +35.1% | +24.1% |
| GPT-OSS 120B | -24.1% | +15.4% | -10.1% |
| Qwen3 235B | -13.8% | -11.7% | -10.3% |
| DeepSeek V3 | +5.0% | +42.9% | +39.7% |

Результаты: `experiments/results/`

### Google Colab (T4 GPU)

Baseline:

| ASR | Датасет | Язык | Mean WER |
|-----|---------|------|----------|
| Whisper base | LibriSpeech test-clean | EN | 5.69% |
| Whisper base | FLEURS | RU | 22.26% |
| GigaAM v2 CTC | FLEURS | RU | 9.84% |
| Wav2Vec2 XLS-R 1B | FLEURS | RU | 17.80% |

LLM-коррекция (4 ASR x 5 LLM = 20 экспериментов):

| LLM | Whisper EN | Whisper RU | GigaAM RU | W2V2 RU |
|-----|-----------|-----------|-----------|---------|
| Qwen2.5 7B | -17.7% | +4.6% | +2.1% | +12.1% |
| Qwen2.5 72B | -15.7% | +36.9% | +17.9% | +27.2% |
| GPT-OSS 120B | -29.1% | +5.0% | -3.2% | +15.0% |
| Qwen3 235B | -8.2% | -1.6% | -24.6% | -0.9% |
| DeepSeek V3 | -2.6% | +36.4% | +26.8% | +35.2% |

Ноутбук: `notebooks/02_llm_benchmark_demo.ipynb`

### Лучшие модели (оба запуска)

| LLM | Whisper RU | W2V2 RU | Вердикт |
|-----|-----------|---------|---------|
| DeepSeek V3 (685B) | +36 -- +43% | +35 -- +40% | Лучшая модель |
| Qwen2.5 72B | +35 -- +37% | +24 -- +27% | Стабильный второй |
| GPT-OSS 120B | +5 -- +15% | -10 -- +15% | Нестабильная |
| Qwen3 235B | -12 -- -2% | -10 -- -1% | Не рекомендуется |

Небольшие расхождения между средами объясняются различиями в точности
вычислений (fp16 на GPU vs fp32 на CPU) и недетерминизмом LLM API.

### Ключевые выводы

1. LLM-коррекция стабильно улучшает WER при baseline > 15%.
   При baseline < 10% модели чаще ухудшают результат.

2. DeepSeek V3 - единственная модель, улучшающая все эксперименты
   (включая английский) в локальном запуске.

3. Dense-модели (Qwen2.5 72B) стабильнее MoE (Qwen3 235B, GPT-OSS 120B)
   для задачи ASR-коррекции.

4. Reasoning-модели (Qwen3 235B) не подходят для простой текстовой
   коррекции - генерируют объяснения вместо исправлений.

## Тестирование

Проект покрыт 42 unit-тестами по 12 модулям:

| Модуль | Тесты | Что проверяется |
|--------|-------|-----------------|
| test_audio | 2 | Сохранение WAV, непустой файл |
| test_baseline_cache | 3 | Кеширование baseline-результатов |
| test_benchmark_io | 2 | Создание директорий, запись CSV |
| test_benchmark_summary | 3 | Сводная таблица, пустые данные, деградация |
| test_clean_response | 5 | Постобработка ответов LLM |
| test_config | 3 | Значения по умолчанию, устройство, переопределение |
| test_llm_client | 4 | Клиент HF API, fallback при ошибках |
| test_metrics | 6 | WER, CER: точное совпадение, ошибки, пустые строки |
| test_normalize | 5 | Нормализация текста: регистр, пунктуация, пробелы |
| test_prompts | 3 | Промпты EN/RU, язык по умолчанию |
| test_save_samples | 3 | Сохранение аудио-примеров, лимит, пропуск существующих |
| test_viz_common | 3 | Построение DataFrame для визуализации |

Запуск:

```bash
python -m pytest tests/ -v
```

## Установка

### Системные требования

- Python 3.10+
- ffmpeg
- GPU не обязателен (Whisper и Wav2Vec2 работают на CPU)

### Настройка окружения

```bash
git clone https://github.com/alexshevvv/llm-asr-correction.git
cd llm-asr-correction
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp env.example .env
# Заполнить .env: HF_TOKEN=hf_your_token_here
```

### Получение API-ключа

HuggingFace Token (бесплатно или Pro для расширенных лимитов):
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Запуск

### Полный бенчмарк

```bash
python -m scripts.run_benchmark
```

Пайплайн выполняет:
1. Загрузку датасетов (LibriSpeech EN, FLEURS RU)
2. ASR baseline (Whisper base + Wav2Vec2 XLS-R)
3. LLM-коррекцию через HuggingFace Inference API (5 моделей)
4. Сохранение CSV-результатов и JSON-конфига
5. Генерацию 7 графиков (seaborn + matplotlib)

Первый запуск: ~30 минут (ASR baseline + LLM API).
Повторные запуски: ~20 минут (baseline из кеша `data/processed/`).

### Тесты

```bash
python -m pytest tests/ -v
```

## Визуализация

Бенчмарк автоматически генерирует 7 графиков в `experiments/results/`:

| График | Файл | Описание |
|--------|------|----------|
| WER Change | plot_wer_change.png | Grouped barplot по ASR и LLM |
| Baseline vs Corrected | plot_corrected_wer.png | Faceted сравнение до/после |
| Diverging bar | plot_diverging.png | Все эксперименты: улучшение/деградация |
| Heatmap | plot_heatmap.png | Матрица WER Change (RdYlGn) |
| Scatter | plot_scatter.png | Baseline WER vs эффективность коррекции |
| ASR Baseline | plot_baselines.png | Сравнение ASR-моделей |
| Radar | plot_radar.png | Профиль каждой LLM |

## Дорожная карта

- [x] Baseline ASR: Whisper + GigaAM + Wav2Vec2 на EN/RU датасетах
- [x] LLM-коррекция: 5 моделей (7B -- 685B) через HuggingFace Inference API
- [x] Модуль визуализации (7 типов графиков)
- [x] Кеширование baseline-результатов
- [x] Постобработка ответов LLM (clean_response)
- [x] 42 unit-теста
- [ ] Confidence-aware коррекция
- [ ] Оптимизация промптов (few-shot, domain-specific)
- [ ] Масштабирование до 500--1000 сэмплов
- [ ] Анализ по типам ошибок (замены, вставки, удаления)

## Автор

**Алексей Шевченко** --- ВШЭ Нижний Новгород

## Лицензия

MIT License