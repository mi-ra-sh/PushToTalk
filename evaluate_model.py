"""
Оцінка якості Whisper: базова модель vs fine-tuned з LoRA.
Міряє WER загальний та по категоріях: [Щ], [MIX], [UA], [IT], [NUM]
"""

import sys
import os
import csv
import json
import time
import argparse
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ============ НАЛАШТУВАННЯ ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "training_data")
USER_AUDIO_DIR = os.path.join(USER_DATA_DIR, "audio")
USER_METADATA = os.path.join(USER_DATA_DIR, "metadata.csv")
ADAPTER_DIR = os.path.join(BASE_DIR, "whisper_lora_adapter")
REPORT_FILE = os.path.join(BASE_DIR, "evaluation_report.txt")

MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "uk"
# ======================================


def load_base_model(device="cuda"):
    """Завантажити базову модель"""
    print("  Завантаження базової моделі...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
    return model, processor


def load_finetuned_model(device="cuda"):
    """Завантажити fine-tuned модель з LoRA адаптером"""
    print("  Завантаження fine-tuned моделі з LoRA...")
    processor = WhisperProcessor.from_pretrained(ADAPTER_DIR, language=LANGUAGE, task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()  # Merge LoRA для швидшого inference
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
    return model, processor


def transcribe(model, processor, audio_path, device="cuda"):
    """Транскрибувати один файл"""
    audio, sr = sf.read(audio_path)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=LANGUAGE,
            task="transcribe",
            max_new_tokens=225,
        )

    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text


def load_eval_data(metadata_path, audio_dir, max_samples=None):
    """Завантажити дані для оцінки"""
    samples = []
    if not os.path.exists(metadata_path):
        return samples

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            audio_path = os.path.join(audio_dir, row['file_name'])
            if os.path.exists(audio_path):
                samples.append({
                    'audio_path': audio_path,
                    'reference': row['transcription'],
                    'category': row.get('category', 'UNKNOWN'),
                    'file_name': row['file_name'],
                })

    if max_samples and len(samples) > max_samples:
        # Рівномірно з кожної категорії
        by_cat = defaultdict(list)
        for s in samples:
            by_cat[s['category']].append(s)

        per_cat = max(1, max_samples // len(by_cat))
        selected = []
        for cat, cat_samples in by_cat.items():
            selected.extend(cat_samples[:per_cat])
        samples = selected[:max_samples]

    return samples


def evaluate_model(model, processor, samples, model_name="model"):
    """Оцінити модель на зразках"""
    results = []
    categories = defaultdict(lambda: {"refs": [], "hyps": [], "times": []})
    all_refs = []
    all_hyps = []

    print(f"\n  Оцінка {model_name}: {len(samples)} зразків")

    for i, sample in enumerate(samples):
        t_start = time.time()
        hypothesis = transcribe(model, processor, sample['audio_path'])
        t_elapsed = time.time() - t_start

        reference = sample['reference']
        cat = sample['category']

        all_refs.append(reference)
        all_hyps.append(hypothesis)
        categories[cat]["refs"].append(reference)
        categories[cat]["hyps"].append(hypothesis)
        categories[cat]["times"].append(t_elapsed)

        results.append({
            'file': sample['file_name'],
            'category': cat,
            'reference': reference,
            'hypothesis': hypothesis,
            'time': t_elapsed,
            'match': reference.lower().strip() == hypothesis.lower().strip(),
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"    [{i+1}/{len(samples)}] {t_elapsed:.2f}с")

    # Загальний WER
    overall_wer = compute_wer(all_refs, all_hyps) * 100

    # WER по категоріях
    cat_wers = {}
    for cat, data in sorted(categories.items()):
        cat_wer = compute_wer(data["refs"], data["hyps"]) * 100
        avg_time = np.mean(data["times"])
        cat_wers[cat] = {
            "wer": cat_wer,
            "count": len(data["refs"]),
            "avg_time": avg_time,
        }

    return {
        "model_name": model_name,
        "overall_wer": overall_wer,
        "category_wers": cat_wers,
        "results": results,
        "total_samples": len(samples),
    }


def has_shch_word(text):
    """Перевірити чи є слова з щ"""
    shch_words = ["ще", "щось", "щоб", "щодо", "що", "навіщо", "нащо", "щасливий",
                  "щастя", "щирий", "щедрий", "щоденно", "щільний", "ніщо", "щітка",
                  "щойно", "щоправда", "нещодавно", "площа", "борщ", "дощ", "прощай"]
    lower = text.lower()
    return any(w in lower for w in shch_words)


def print_comparison(base_eval, ft_eval, report_file=None):
    """Вивести порівняння базової і fine-tuned моделі"""
    lines = []

    def p(text=""):
        lines.append(text)
        print(text)

    p("=" * 70)
    p("  РЕЗУЛЬТАТИ ОЦІНКИ: БАЗОВА vs FINE-TUNED")
    p("=" * 70)

    p(f"\n  {'Метрика':<30} {'Базова':>15} {'Fine-tuned':>15} {'Різниця':>10}")
    p("-" * 70)
    p(f"  {'Загальний WER':<30} {base_eval['overall_wer']:>14.1f}% {ft_eval['overall_wer']:>14.1f}% {ft_eval['overall_wer'] - base_eval['overall_wer']:>+9.1f}%")

    p(f"\n  WER по категоріях:")
    p("-" * 70)
    all_cats = sorted(set(list(base_eval['category_wers'].keys()) + list(ft_eval['category_wers'].keys())))
    for cat in all_cats:
        base_wer = base_eval['category_wers'].get(cat, {}).get('wer', -1)
        ft_wer = ft_eval['category_wers'].get(cat, {}).get('wer', -1)
        count = base_eval['category_wers'].get(cat, {}).get('count', 0)
        base_str = f"{base_wer:.1f}%" if base_wer >= 0 else "N/A"
        ft_str = f"{ft_wer:.1f}%" if ft_wer >= 0 else "N/A"
        diff_str = f"{ft_wer - base_wer:+.1f}%" if base_wer >= 0 and ft_wer >= 0 else "N/A"
        p(f"  [{cat:<5}] ({count:>3} зр.)  {base_str:>15} {ft_str:>15} {diff_str:>10}")

    # Конкретні приклади
    p(f"\n  Приклади транскрипцій (перші 20):")
    p("-" * 70)
    for i, (base_r, ft_r) in enumerate(zip(base_eval['results'][:20], ft_eval['results'][:20])):
        ref = base_r['reference']
        base_hyp = base_r['hypothesis']
        ft_hyp = ft_r['hypothesis']

        # Позначки
        base_ok = "OK" if base_r['match'] else "XX"
        ft_ok = "OK" if ft_r['match'] else "XX"

        if base_hyp != ft_hyp:  # Показувати тільки де є різниця
            p(f"\n  [{base_r['category']}] {base_r['file']}")
            p(f"    REF:  {ref}")
            p(f"    BASE [{base_ok}]: {base_hyp}")
            p(f"    FINE [{ft_ok}]:  {ft_hyp}")

    # Щ-слова окремо
    p(f"\n  Тест щ-слів:")
    p("-" * 70)
    for base_r, ft_r in zip(base_eval['results'], ft_eval['results']):
        if has_shch_word(base_r['reference']):
            ref = base_r['reference']
            base_hyp = base_r['hypothesis']
            ft_hyp = ft_r['hypothesis']
            if base_hyp != ft_hyp:
                p(f"    REF:  {ref}")
                p(f"    BASE: {base_hyp}")
                p(f"    FINE: {ft_hyp}")
                p()

    p("\n" + "=" * 70)

    # Зберегти звіт
    if report_file:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        p(f"\n  Звіт збережено: {report_file}")


def print_single(eval_result, report_file=None):
    """Вивести результати одної моделі"""
    lines = []

    def p(text=""):
        lines.append(text)
        print(text)

    p("=" * 70)
    p(f"  РЕЗУЛЬТАТИ: {eval_result['model_name']}")
    p("=" * 70)

    p(f"\n  Загальний WER: {eval_result['overall_wer']:.1f}%")
    p(f"  Зразків: {eval_result['total_samples']}")

    p(f"\n  WER по категоріях:")
    p("-" * 50)
    for cat, data in sorted(eval_result['category_wers'].items()):
        p(f"  [{cat:<5}] ({data['count']:>3} зр.) WER: {data['wer']:.1f}%  avg time: {data['avg_time']:.2f}с")

    p(f"\n  Приклади помилок:")
    p("-" * 50)
    errors = [r for r in eval_result['results'] if not r['match']][:15]
    for r in errors:
        p(f"  [{r['category']}] {r['file']}")
        p(f"    REF:  {r['reference']}")
        p(f"    HYP:  {r['hypothesis']}")
        p()

    p("=" * 70)

    if report_file:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        p(f"\n  Звіт збережено: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper base vs fine-tuned")
    parser.add_argument("--base-only", action="store_true", help="Evaluate only base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Evaluate only fine-tuned model")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    if not os.path.exists(USER_METADATA):
        print(f"Немає даних для оцінки: {USER_METADATA}")
        print("Запишіть дані через collect_training_data.py")
        sys.exit(1)

    samples = load_eval_data(USER_METADATA, USER_AUDIO_DIR, max_samples=args.max_samples)
    print(f"Завантажено {len(samples)} зразків для оцінки")

    if len(samples) == 0:
        print("Немає валідних зразків!")
        sys.exit(1)

    has_adapter = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))

    if args.finetuned_only:
        if not has_adapter:
            print(f"Адаптер не знайдено: {ADAPTER_DIR}")
            sys.exit(1)
        model, processor = load_finetuned_model()
        ft_eval = evaluate_model(model, processor, samples, "Fine-tuned")
        print_single(ft_eval, REPORT_FILE)
    elif args.base_only or not has_adapter:
        if not has_adapter:
            print(f"  Адаптер не знайдено, оцінюю лише базову модель")
        model, processor = load_base_model()
        base_eval = evaluate_model(model, processor, samples, "Base")
        print_single(base_eval, REPORT_FILE)
    else:
        # Порівняння
        base_model, base_proc = load_base_model()
        base_eval = evaluate_model(base_model, base_proc, samples, "Base")

        # Звільнити пам'ять
        del base_model
        torch.cuda.empty_cache()

        ft_model, ft_proc = load_finetuned_model()
        ft_eval = evaluate_model(ft_model, ft_proc, samples, "Fine-tuned")

        print_comparison(base_eval, ft_eval, REPORT_FILE)


if __name__ == "__main__":
    main()
