# LLM-ASR-Correction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/inference-providers)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Tests](https://img.shields.io/badge/tests-94_passed-brightgreen?logo=pytest&logoColor=white)]()

[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13+-4C72B0)](https://seaborn.pydata.org/)
[![jiwer](https://img.shields.io/badge/jiwer-3.0+-blue)](https://github.com/jitsi/jiwer)
[![Transformers](https://img.shields.io/badge/Transformers-4.45+-orange?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)

Распознавание речи с корректировкой ошибок с использованием больших языковых моделей

*Дипломная работа --- ВШЭ Нижний Новгород, 2026*

---

## Описание

Исследование эффективности больших языковых моделей (LLM) для постобработки
и коррекции ошибок автоматического распознавания речи (ASR).
Проект реализует сравнительный бенчмарк на матрице
**6 ASR x 6 LLM x 5 датасетов** для английского и русского языков.

### Главный тезис

Предлагается двухкомпонентный метод LLM-коррекции ASR:
**(1)** confidence-aware вызов --- применение LLM только к низкоуверенным фрагментам,
**(2)** фонетическая фильтрация замен --- защита от галлюцинаций LLM.
Метод оценивается на сравнительном бенчмарке с детализацией по типам ошибок
и устойчивостью к шуму.

## Организация проекта

Проект построен как научно-исследовательская работа с двумя параллельными слоями:

- **Локальный проект (PyCharm)** --- модульный код в `src/` и `scripts/`,
  покрытый тестами. ASR-модели Whisper base и Wav2Vec2 XLS-R 1B работают на CPU.
  LLM-коррекция через HuggingFace Inference API.
  Результаты сохраняются в `experiments/results/`.

- **Google Colab** --- ноутбук `notebooks/03_llm_benchmark_expanded.ipynb`
  с расширенным набором ASR-моделей (6 штук, включая GigaAM и Whisper medium,
  требующие CUDA). Ноутбук использует те же LLM через HuggingFace Inference API
  и формирует результаты на полной матрице экспериментов.

Оба слоя дополняют друг друга: локальный проект обеспечивает воспроизводимость
и тестируемость, Colab --- доступ к GPU-моделям.

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
│   └── 03_llm_benchmark_expanded.ipynb  # Полный бенчмарк (6 ASR x 6 LLM x 5 DS)
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
│   │   ├── whisper_transcribe.py   # OpenAI Whisper
│   │   ├── wav2vec2_transcribe.py  # Wav2Vec2 XLS-R 1B
│   │   ├── gigaam_transcribe.py    # Sber GigaAM (только Colab)
│   │   ├── registry_data.py        # Метаданные ASR-моделей
│   │   └── registry_query.py       # Запросы к ASR registry
│   ├── correction/                 # LLM-коррекция
│   │   ├── llm_client.py           # HuggingFace Inference API клиент
│   │   ├── clean_response.py       # Постобработка ответов LLM
│   │   ├── prompts.py              # Системные и пользовательские промпты
│   │   ├── llm_registry_data.py    # Метаданные LLM-моделей
│   │   └── llm_registry_query.py   # Запросы к LLM registry
│   ├── evaluation/                 # Метрики
│   │   ├── metrics.py              # WER, CER
│   │   └── normalize.py            # Нормализация текста
│   ├── visualization/              # Визуализация (6 типов графиков)
│   │   ├── common.py               # ASR_GROUPS, общие настройки
│   │   ├── analysis.py             # Построение analysis_df
│   │   ├── wer_change.py           # Grouped barplot по Dataset
│   │   ├── heatmap.py              # Heatmap по ASR-семействам
│   │   ├── scatter.py              # Baseline WER vs эффективность
│   │   ├── radar.py                # Radar по Dataset
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
├── tests/                          # Unit-тесты (94 теста)
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
│   ├── test_llm_client.py          # HF API клиент, fallback при ошибках
│   ├── test_llm_registry_data.py   # Метаданные LLM registry
│   ├── test_llm_registry_query.py  # Запросы и фильтрация LLM
│   ├── test_memory.py              # Мониторинг памяти
│   ├── test_metrics.py             # WER, CER: точные, ошибочные, пустые
│   ├── test_normalize.py           # Нормализация: регистр, пунктуация
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
| DeepSeek V3 | 685B MoE | MoE | DeepSeek |

## Датасеты

| Название | Язык | Стиль | Описание |
|----------|------|-------|----------|
| LibriSpeech test-clean | EN | clean | Стандартный бенчмарк ASR, чистая студийная речь |
| LibriSpeech test-other | EN | noisy | Тот же корпус, с шумом и вариациями произношения |
| FLEURS English | EN | clean | Корпус Google с носителями языка |
| FLEURS Russian | RU | clean | Чистая русская речь из корпуса FLEURS |
| SOVA audiobooks | RU | literary | Профессиональная литературная русская речь |

## Результаты

### Локальный запуск (2 ASR x 6 LLM x 5 датасетов, 30 сэмплов)

Baseline WER:

| ASR | Dataset | Baseline WER |
|-----|---------|-------------|
| Whisper base | LibriSpeech test-clean (EN) | 4.61% |
| Whisper base | LibriSpeech test-other (EN) | 10.04% |
| Whisper base | FLEURS English | 13.73% |
| Whisper base | FLEURS Russian | 25.68% |
| Whisper base | SOVA audiobooks (RU) | 43.82% |
| Wav2Vec2 XLS-R 1B | FLEURS Russian | 19.03% |
| Wav2Vec2 XLS-R 1B | SOVA audiobooks (RU) | 20.96% |

WER Change (%) после LLM-коррекции (положительные = улучшение):

| LLM | WB / LS-clean | WB / LS-other | WB / FLEURS EN | WB / FLEURS RU | WB / SOVA | W2V2 / FLEURS RU | W2V2 / SOVA |
|-----|--------------|--------------|---------------|---------------|----------|-----------------|------------|
| Qwen2.5 7B | -48.1 | -26.5 | -1.1 | +3.4 | +3.5 | +15.9 | -11.7 |
| Llama 3.3 70B | -82.2 | -43.3 | -23.2 | +33.1 | -329.7 | +33.4 | -19.8 |
| Qwen2.5 72B | -48.2 | -15.9 | +9.2 | +36.8 | +3.3 | +28.7 | +9.7 |
| GPT-OSS 120B | -83.4 | -11.4 | +1.4 | +2.4 | -9.3 | +15.6 | -8.3 |
| Qwen3 235B | -24.2 | -8.3 | -13.0 | -17.7 | +3.2 | -2.2 | -2.1 |
| DeepSeek V3 | -30.6 | -11.1 | +14.0 | +36.2 | +8.4 | +21.3 | +7.0 |

### Google Colab (6 ASR x 6 LLM x 5 датасетов, 30 сэмплов)

114 экспериментов. Полные результаты --- в ноутбуке `notebooks/03_llm_benchmark_expanded.ipynb`.

Лучшие модели по русскому языку (средний WER Change по 10 RU-экспериментам):

| LLM | Средний WER Change (RU) | Стабильность |
|-----|------------------------|-------------|
| Qwen3 235B | +22.3% | 10/10 улучшений |
| Qwen2.5 72B | +17.3% | 10/10 улучшений |
| DeepSeek V3 | +10.0% | 7/10 улучшений |
| Llama 3.3 70B | +7.1% | 6/10 улучшений |
| Qwen2.5 7B | +4.2% | 5/10 улучшений |
| GPT-OSS 120B | -12.3% | 3/10 улучшений |

### Ключевые выводы

1. **LLM-коррекция стабильно помогает при baseline WER выше 15%.**
   При baseline ниже 10% модели чаще ухудшают результат.

2. **Лучшая модель для русского --- Qwen2.5 72B (dense).**
   Стабильно улучшает все 10 русских экспериментов.

3. **Dense-модели стабильнее MoE** для задачи ASR-коррекции.

4. **Wav2Vec2 XLS-R 1B EN --- наиболее благоприятен для LLM-коррекции.**
   CTC-модель без языковой модели допускает лингвистические ошибки,
   которые LLM исправляет эффективно.

5. **DeepSeek V3 генерирует галлюцинации на английском.**
   Добавляет пояснительный текст вместо коррекции. На русском работает адекватно.
   Реализована система clean_response с детекцией отказов и удалением
   мета-комментариев для защиты от этой проблемы.

## Тестирование

94 unit-теста по 19 модулям:

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
2. ASR baseline (Whisper base + Wav2Vec2 XLS-R 1B)
3. LLM-коррекцию через HuggingFace Inference API (6 моделей)
4. Сохранение analysis CSV + per-correction CSV + JSON-конфига
5. Генерацию 8 графиков (seaborn + matplotlib)
6. Логирование в `experiments/logs/`

Первый запуск: ~20 минут (ASR baseline + LLM API).

### Тесты

```bash
python -m pytest tests/ -v
```

## Визуализация

Бенчмарк генерирует 8 графиков в `experiments/results/`:

| График | Файл | Описание |
|--------|------|----------|
| WER Change | plot_wer_change.png | Средний WER Change по LLM и Dataset |
| Heatmap (Whisper base) | plot_heatmap_whisper_base.png | Матрица WER Change для Whisper base |
| Heatmap (Wav2Vec2) | plot_heatmap_wav2vec2_family.png | Матрица WER Change для Wav2Vec2 |
| Scatter | plot_scatter.png | Baseline WER vs эффективность коррекции |
| Radar | plot_radar.png | Профиль каждой LLM по датасетам |
| Stacked (Whisper) | plot_stacked_whisper_base.png | Improved/Degraded/Unchanged |
| Stacked (Wav2Vec2) | plot_stacked_wav2vec2_family.png | Improved/Degraded/Unchanged |
| Baseline | plot_baselines.png | Сравнение ASR baseline WER |

## Дорожная карта

### Выполнено

- [x] Baseline ASR: 6 моделей (Whisper base/medium, GigaAM CTC/RNNT, Wav2Vec2 RU/EN)
- [x] LLM-коррекция: 6 моделей (7B -- 685B) через HuggingFace Inference API
- [x] 5 датасетов (3 EN + 2 RU): LibriSpeech, FLEURS, SOVA
- [x] Registry-архитектура для ASR, LLM и датасетов
- [x] Модуль визуализации (6 типов графиков, разбивка по ASR-семействам)
- [x] Постобработка ответов LLM (clean_response: refusal detection, tail removal)
- [x] Сохранение per-correction CSV для анализа галлюцинаций
- [x] 94 unit-теста

### Приоритет 1: двухкомпонентный метод

- [ ] **Confidence-aware вызов LLM** --- token-level logprobs из Whisper,
      LLM-коррекция только для фрагментов с низким confidence
- [ ] **Фонетическая фильтрация замен** --- post-hoc фильтр на выходе LLM,
      откат замен слишком далёких от оригинала по расстоянию Левенштейна

### Приоритет 2: углубление анализа

- [ ] **Классификация ошибок** --- разбивка WER на substitution/insertion/deletion
      через jiwer.process_words()
- [ ] **Сравнение промптов** --- текущий простой промпт vs TAP-стиль
      (Task-Activating Prompting с few-shot примерами)

### Приоритет 3: расширение экспериментов

- [ ] **Эксперименты с шумом** --- AWGN + структурированные помехи
      (фоновая речь, транспорт) на разных SNR уровнях
- [ ] **N-best гипотезы** --- beam search из Whisper для прокси-метрики confidence

## Автор

**Алексей Шевченко** --- ВШЭ Нижний Новгород
