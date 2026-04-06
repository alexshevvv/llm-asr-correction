# experiments/

Артефакты запусков бенчмарка. В отличие от `data/`, содержимое
этой папки **частично коммитится** в репозиторий: финальные результаты
сохраняются как референс для воспроизводимости, локальные логи —
игнорируются.

## Структура

### configs/

JSON-конфиги запусков бенчмарка. Создаются автоматически функцией
`save_run_config()` из `scripts/benchmark_io.py` при каждом вызове
`run_benchmark.py`.

Файл именуется по timestamp: `run_YYYYMMDD_HHMMSS.json`.
Содержит: устройство (CPU/CUDA), модель Whisper, число сэмплов на
датасет, список LLM моделей, краткую статистику по baseline (samples,
mean_wer, errors).

Эти файлы **коммитятся** в git и служат свидетельством того, какая
конфигурация дала какие результаты.

### logs/

Логи запусков (`run_YYYYMMDD_HHMMSS.log`). Создаются модулем `logging`
при запуске `run_benchmark.py`.

Эти файлы **не коммитятся** (см. `.gitignore`): они полезны локально
для отладки, но не нужны в репозитории.

### results/

Финальные результаты бенчмарка. Все файлы здесь **коммитятся** в git.

**Baseline ASR (3 файла):**
- `baseline_whisper_en.csv` --- Whisper base на LibriSpeech (английский)
- `baseline_whisper_ru.csv` --- Whisper base на FLEURS (русский)
- `baseline_w2v2_ru.csv` --- Wav2Vec2 XLS-R на FLEURS (русский)

**LLM-коррекция (5 LLM × 3 ASR = 15 файлов):**
- `correction_<LLM>__<ASR>.csv` для каждой пары
- LLM: Qwen2.5 7B/72B, GPT-OSS 120B, Qwen3 235B, DeepSeek V3
- ASR: whisper_en, whisper_ru, w2v2_ru

**Сводка:**
- `benchmark_summary.csv` --- агрегированная таблица WER change
  по всем экспериментам

**Графики (7 PNG):**
- `plot_baselines.png` --- сравнение baseline моделей
- `plot_wer_change.png` --- WER change по моделям и ASR
- `plot_corrected_wer.png` --- baseline vs corrected WER (фасеты по LLM)
- `plot_heatmap.png` --- LLM × ASR heatmap WER change
- `plot_diverging.png` --- все эксперименты, отсортированные по эффекту
- `plot_scatter.png` --- baseline WER vs эффективность коррекции
- `plot_radar.png` --- radar chart LLM по ASR категориям

## Как воспроизвести
```bash
python -m scripts.run_benchmark
```

Запуск создаст новые файлы в `configs/`, `logs/`, `results/` поверх
существующих. Старые версии не удаляются автоматически --- если нужен
чистый запуск, удалите содержимое вручную:
```bash
rm experiments/configs/*.json
rm experiments/logs/*.log
rm experiments/results/*
python -m scripts.run_benchmark
```