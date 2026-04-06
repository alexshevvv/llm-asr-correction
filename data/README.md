# data/

Данные проекта. Содержимое этой папки не коммитится в репозиторий
(см. `.gitignore`). Все данные создаются автоматически при запуске бенчмарка.

## Структура

### raw/

Кеш датасетов HuggingFace. Создается автоматически при первом запуске.
Содержит lock-файлы LibriSpeech (EN) и Google FLEURS (RU).

### processed/

Кешированные результаты ASR baseline. Позволяет повторно запускать
LLM-коррекцию без пересчета ASR (экономия ~10 минут на CPU).

Файлы:
- `baseline_whisper_en.csv` --- Whisper base на LibriSpeech
- `baseline_whisper_ru.csv` --- Whisper base на FLEURS
- `baseline_w2v2_ru.csv` --- Wav2Vec2 XLS-R на FLEURS

### samples/

Аудио-примеры из датасетов (по 3 на язык). Сохраняются при первом
запуске для документации и ручной проверки.

## Как заполнить
```bash
python -m scripts.run_benchmark
```

Для пересчета baseline с нуля:
```bash
rm data/processed/*.csv
python -m scripts.run_benchmark
```