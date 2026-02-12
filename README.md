# LLM-ASR-Correction

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Распознавание речи с корректировкой ошибок с использованием больших языковых моделей**

*Дипломная работа — ВШЭ Нижний Новгород, 2026*

[Описание](#описание)

</div>

---

## Статус проекта

> **DEMO — В РАЗРАБОТКЕ**
> 
> Проект находится на этапе активной разработки. API и структура могут меняться.

| Этап | Статус |
|------|--------|
| 1. Исследование ASR моделей | Выполнено |
| 2. Подбор датасетов | Выполнено |
| 3. Baseline эксперименты (WER) | В работе |
| 4. LLM коррекция ошибок | Запланировано |

---

## Описание

Исследование эффективности использования **больших языковых моделей** (GPT-4, Claude) для постобработки и коррекции ошибок в выходных данных систем автоматического распознавания речи (ASR).

### Гипотеза

LLM способны исправлять типичные ошибки ASR (омофоны, редкие слова, контекстные ошибки), снижая WER на 10-30%.

---

## Технологии

<div align="center">

### ASR Models
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat-square&logo=openai&logoColor=white)
![GigaAM](https://img.shields.io/badge/Sber-GigaAM-21A038?style=flat-square)

### LLM Correction
![GPT-4](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai&logoColor=white)
![Claude](https://img.shields.io/badge/Anthropic-Claude-D4A574?style=flat-square)

### Stack
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)

</div>

---

## Модели ASR

| Модель | Язык | Параметры | Описание |
|--------|------|-----------|----------|
| **Whisper large-v3** | Multilingual | 1.5B | OpenAI, SOTA multilingual ASR |
| **GigaAM-v3** | Russian | - | SberDevices, SOTA для русского |

---

## Датасеты

### Русский язык

| Датасет | Размер | Описание |
|---------|--------|----------|
| **Golos** | ~1240 ч | Сбер, разнообразная речь |
| **Russian LibriSpeech** | ~98 ч | Аудиокниги |
| **Common Voice RU** | ~200+ ч | Краудсорсинг Mozilla |

### Английский язык

| Датасет | Размер | Описание |
|---------|--------|----------|
| **LibriSpeech** | ~960 ч | Аудиокниги |
| **Common Voice EN** | ~2000+ ч | Краудсорсинг Mozilla |

---

## Структура проекта

```
llm-asr-correction/
├── src/
│   ├── asr/                       # ASR транскрибаторы
│   │   ├── base.py                # Базовый класс
│   │   ├── whisper_transcriber.py # Whisper
│   │   └── gigaam_transcriber.py  # GigaAM
│   ├── correction/                # LLM корректоры
│   ├── evaluation/                # Метрики
│   │   └── metrics.py             # WER, CER
│   └── utils/                     # Утилиты
│       └── config.py              # Конфигурация
├── data/                          # Датасеты (не в git)
├── experiments/                   # Результаты
├── notebooks/                     # Jupyter ноутбуки
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Автор

**Алексей Шевченко**

Студент 3 курса, Программная инженерия  
НИУ ВШЭ — Нижний Новгород

[![GitHub](https://img.shields.io/badge/GitHub-alexshevvv-181717?style=flat-square&logo=github)](https://github.com/alexshevvv)