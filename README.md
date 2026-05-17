# LLM-ASR-Correction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/inference-providers)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Tests](https://img.shields.io/badge/tests-113_passed-brightgreen?logo=pytest&logoColor=white)]()

[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13+-4C72B0)](https://seaborn.pydata.org/)
[![jiwer](https://img.shields.io/badge/jiwer-3.0+-blue)](https://github.com/jitsi/jiwer)
[![Transformers](https://img.shields.io/badge/Transformers-4.45+-orange?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)

Распознавание речи с корректировкой ошибок с помощью больших языковых моделей

*Дипломная работа - ВШЭ Нижний Новгород, 2026*

---

## Описание

Исследование эффективности больших языковых моделей (LLM) для постобработки
и коррекции ошибок автоматического распознавания речи (ASR).
Проект реализует сравнительный бенчмарк на матрице
**6 ASR × 6 LLM × 5 датасетов** для английского и русского языков.

### Главный тезис

Предлагается двухкомпонентный метод LLM-коррекции ASR:
**(1)** confidence-aware вызов - применение LLM только к низкоуверенным фрагментам
(avg_logprob Whisper, порог -0.35),
**(2)** фонетическая фильтрация замен - post-hoc фильтр на основе расстояния
Левенштейна для защиты от галлюцинаций LLM.
Метод оценивается на сравнительном бенчмарке с детализацией по типам ошибок
и абляционным исследованием вклада каждого компонента.

## Организация проекта

Проект построен как научно-исследовательская работа с двумя параллельными слоями:

- **Локальный проект (PyCharm)** - модульный код в `src/` и `scripts/`,
  покрытый 113 тестами. ASR-модели Whisper base и Wav2Vec2 XLS-R 1B работают на CPU.
  LLM-коррекция через HuggingFace Inference API.
  Результаты сохраняются в `experiments/results/`.

- **Google Colab** - ноутбук `notebooks/04_llm_benchmark_final.ipynb`
  с расширенным набором ASR-моделей (6 штук, включая GigaAM и Whisper medium,
  требующие CUDA). Ноутбук формирует результаты на полной матрице экспериментов
  и содержит абляционное исследование.

Оба слоя дополняют друг друга: локальный проект обеспечивает воспроизводимость
и тестируемость, Colab - доступ к GPU-моделям и полный набор экспериментов.

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
│   ├── 02_llm_benchmark_demo.ipynb # Второй прототип бенчмарка (5 LLM x 4 ASR)
│   ├── 03_llm_benchmark_expanded.ipynb  # Полный бенчмарк (6 ASR x 6 LLM x 5 DS)
│   └── 04_llm_benchmark_final.ipynb     # Финальный: confidence-aware + фильтр + ablation
│
├── scripts/                        # Скрипты запуска экспериментов
│   ├── run_benchmark_v2.py         # Точка входа: полный pipeline
│   ├── benchmark_baselines_v2.py   # Registry-driven baseline ASR matrix
│   ├── benchmark_correction_v2.py  # Registry-driven LLM correction matrix
│   └── benchmark_save.py           # Сохранение CSV и JSON-конфигов
│
├── src/                            # Исходный код (модули)
│   ├── asr/                        # ASR-модели
│   │   ├── base.py                 # Абстрактный интерфейс ASR
│   │   ├── whisper_transcribe.py   # OpenAI Whisper (+ confidence extraction)
│   │   ├── wav2vec2_transcribe.py  # Wav2Vec2 XLS-R 1B
│   │   ├── gigaam_transcribe.py    # Sber GigaAM (только Colab)
│   │   ├── registry_data.py        # Метаданные ASR-моделей
│   │   └── registry_query.py       # Запросы к ASR registry
│   ├── correction/                 # LLM-коррекция
│   │   ├── llm_client.py           # HuggingFace Inference API клиент
│   │   ├── clean_response.py       # Постобработка ответов LLM
│   │   ├── levenshtein.py          # Расстояние Левенштейна (символьный уровень)
│   │   ├── phonetic_filter.py      # Фонетический фильтр замен LLM
│   │   ├── prompts.py              # Системные и пользовательские промпты
│   │   ├── llm_registry_data.py    # Метаданные LLM-моделей
│   │   └── llm_registry_query.py   # Запросы к LLM registry
│   ├── evaluation/                 # Метрики
│   │   ├── metrics.py              # WER, CER
│   │   ├── error_classification.py # Классификация ошибок: S/I/D
│   │   └── normalize.py            # Нормализация текста
│   ├── visualization/              # Визуализация (8 типов графиков)
│   │   ├── common.py               # ASR_GROUPS, общие настройки
│   │   ├── analysis.py             # Построение analysis_df
│   │   ├── wer_change.py           # Grouped barplot по Dataset
│   │   ├── heatmap.py              # Heatmap по ASR-семействам
│   │   ├── scatter.py              # Baseline WER vs эффективность
│   │   ├── radar.py                # Radar по Dataset
│   │   ├── error_types.py          # Оркестратор графиков типов ошибок
│   │   ├── error_types_build.py    # Построение error_df (S/I/D по сэмплам)
│   │   ├── error_types_draw_baseline.py # Типы ошибок в baseline
│   │   ├── error_types_draw_delta.py    # Дельта ошибок после LLM
│   │   ├── baseline.py             # ASR baseline comparison
│   │   ├── stacked_bar.py          # Improved/Degraded/Unchanged
│   │   ├── stacked_bar_draw.py     # Отрисовка stacked bar
│   │   └── runner.py               # Запуск всех графиков
│   └── utils/                      # Утилиты
│       ├── config.py               # Конфигурация эксперимента
│       ├── datasets/               # Загрузчики датасетов
│       │   ├── librispeech.py      # LibriSpeech test-clean / test-other
│       │   ├── fleurs.py           # FLEURS English / Russian
│       │   └── sova.py             # SOVA audiobooks
│       ├── datasets_registry_data.py  # Метаданные датасетов
│       ├── datasets_registry_query.py # Запросы к datasets registry
│       ├── dataset_loader.py       # Динамическая загрузка по ключу
│       ├── class_loader.py         # Импорт классов по строковому пути
│       ├── audio.py                # WAV I/O, ресемплинг
│       ├── memory.py               # Мониторинг памяти
│       └── save_samples.py         # Сохранение аудио-примеров
│
├── tests/                          # Unit-тесты (113 тестов, 21 модуль)
│   ├── test_analysis.py            # Построение analysis_df
│   ├── test_asr_registry_data.py   # Метаданные ASR registry
│   ├── test_asr_registry_query.py  # Запросы к ASR registry
│   ├── test_audio.py               # Сохранение WAV, ресемплинг
│   ├── test_class_loader.py        # Динамический импорт классов
│   ├── test_clean_response.py      # Постобработка ответов LLM
│   ├── test_config.py              # Конфигурация, устройство, defaults
│   ├── test_dataset_loader.py      # Загрузка датасетов по ключу
│   ├── test_datasets_registry_data.py # Метаданные datasets registry
│   ├── test_datasets_registry_query.py # Запросы и фильтрация датасетов
│   ├── test_error_classification.py # Классификация ошибок S/I/D
│   ├── test_levenshtein.py         # Расстояние Левенштейна
│   ├── test_llm_client.py          # HF API клиент, fallback при ошибках
│   ├── test_llm_registry_data.py   # Метаданные LLM registry
│   ├── test_llm_registry_query.py  # Запросы и фильтрация LLM
│   ├── test_memory.py              # Мониторинг памяти
│   ├── test_metrics.py             # WER, CER: точные, ошибочные, пустые
│   ├── test_normalize.py           # Нормализация: регистр, пунктуация
│   ├── test_phonetic_filter.py     # Фонетический фильтр замен
│   ├── test_prompts.py             # Промпты EN/RU, выбор языка
│   └── test_save_samples.py        # Сохранение аудио, лимит, дубликаты
│
├── .env                            # API-ключи (не коммитится)
├── env.example                     # Шаблон .env
├── requirements.txt                # Зависимости
├── pyproject.toml                  # Конфигурация проекта
└── README.md
```

## ASR-модели

| Модель | Параметры | Архитектура | Языки | Локально | Colab |
|--------|-----------|-------------|-------|----------|-------|
| Whisper base | 74M | Encoder-Decoder | EN, RU | да | да |
| Whisper medium | 769M | Encoder-Decoder | EN, RU | нет | да |
| GigaAM v2 CTC | 240M | CTC (Сбер) | RU | нет | да |
| GigaAM v2 RNNT | 240M | RNN-Transducer (Сбер) | RU | нет | да |
| Wav2Vec2 XLS-R 1B | 1B | CTC (fine-tuned) | RU | да | да |
| Wav2Vec2 XLS-R 1B EN | 1B | CTC (fine-tuned) | EN | нет | да |

## LLM-модели (HuggingFace Inference API)

| Модель | Параметры | Архитектура | Семейство |
|--------|-----------|-------------|-----------|
| Qwen2.5 7B Instruct | 7B | Dense | Alibaba |
| Llama 3.3 70B Instruct | 70B | Dense | Meta |
| Qwen2.5 72B Instruct | 72B | Dense | Alibaba |
| GPT-OSS 120B | 120B MoE (5.1B active) | MoE | OpenAI |
| Qwen3 235B | 235B MoE (22B active) | MoE | Alibaba |
| DeepSeek V3 | 685B MoE (37B active) | MoE | DeepSeek |

## Датасеты

| Название | Язык | Стиль | Описание |
|----------|------|-------|----------|
| LibriSpeech test-clean | EN | clean | Стандартный бенчмарк ASR, чистая студийная речь |
| LibriSpeech test-other | EN | noisy | Тот же корпус, с шумом и вариациями произношения |
| FLEURS English | EN | clean | Корпус Google с носителями языка |
| FLEURS Russian | RU | clean | Чистая русская речь из корпуса FLEURS |
| SOVA audiobooks | RU | literary | Профессиональная литературная русская речь |

## Результаты

Полные результаты экспериментов, включая абляционное исследование двухкомпонентного
метода (confidence-aware + фонетический фильтр), доступны в ноутбуке
[`notebooks/04_llm_benchmark_final.ipynb`](notebooks/04_llm_benchmark_final.ipynb).

Ноутбук содержит: 114 экспериментов (6 ASR × 6 LLM × 5 датасетов, 30 сэмплов),
классификацию ошибок по типам (S/I/D), фонетическую фильтрацию замен,
анализ confidence scores и 5-уровневое абляционное исследование.

## Тестирование

113 unit-тестов по 21 модулю:

```bash
python -m pytest tests/ -v
```

## Установка

### Системные требования

- Python 3.10+
- ffmpeg
- GPU не обязателен (Whisper base и Wav2Vec2 работают на CPU)

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
python -m scripts.run_benchmark_v2
```

Pipeline выполняет:
1. Загрузку датасетов (LibriSpeech EN, FLEURS RU/EN, SOVA RU)
2. ASR baseline с извлечением confidence (Whisper base + Wav2Vec2 XLS-R 1B)
3. Confidence-aware LLM-коррекцию (порог -0.35, пропуск высокоуверенных сэмплов)
4. Фонетическую фильтрацию замен (порог Левенштейна 0.5)
5. Сохранение analysis CSV + per-correction CSV + JSON-конфига
6. Генерацию 10 графиков (seaborn + matplotlib)
7. Логирование в `experiments/logs/`

Первый запуск: ~30 минут (ASR baseline + LLM API).

### Тесты

```bash
python -m pytest tests/ -v
```

## Визуализация

Бенчмарк генерирует 10 графиков в `experiments/results/`:

| График | Файл | Описание |
|--------|------|----------|
| WER Change | plot_wer_change.png | Средний WER Change по LLM и Dataset |
| Heatmap (Whisper) | plot_heatmap_whisper_base.png | Матрица WER Change для Whisper base |
| Heatmap (Wav2Vec2) | plot_heatmap_wav2vec2_family.png | Матрица WER Change для Wav2Vec2 |
| Scatter | plot_scatter.png | Baseline WER vs эффективность коррекции |
| Radar | plot_radar.png | Профиль каждой LLM по датасетам |
| Stacked (Whisper) | plot_stacked_whisper_base.png | Improved/Degraded/Unchanged для Whisper |
| Stacked (Wav2Vec2) | plot_stacked_wav2vec2_family.png | Improved/Degraded/Unchanged для Wav2Vec2 |
| Baseline | plot_baselines.png | Сравнение ASR baseline WER |
| Error Types Baseline | plot_error_types_baseline.png | Распределение S/I/D в baseline |
| Error Types Delta | plot_error_types_delta.png | Изменение S/I/D после LLM-коррекции |

## Автор

**Алексей Шевченко** - ВШЭ Нижний Новгород