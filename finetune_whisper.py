"""
Fine-tuning Whisper large-v3 з LoRA адаптером.
Комбінує: Mozilla Common Voice (UK) + користувацькі записи.
Зберігає адаптер у whisper_lora_adapter/
"""

import sys
import os
import csv
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Fix encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import numpy as np
from torch.utils.data import Dataset
import evaluate
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, Audio, concatenate_datasets

# ============ НАЛАШТУВАННЯ ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "training_data")
USER_AUDIO_DIR = os.path.join(USER_DATA_DIR, "audio")
USER_METADATA = os.path.join(USER_DATA_DIR, "metadata.csv")
ADAPTER_DIR = os.path.join(BASE_DIR, "whisper_lora_adapter")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "whisper_checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "whisper_logs")

MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "uk"
TASK = "transcribe"

# LoRA
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# Training
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
NUM_EPOCHS = 3
WARMUP_STEPS = 500
SAVE_STEPS = 500
EVAL_STEPS = 500
MAX_CV_HOURS = 5  # Обмеження Common Voice (годин)
# ======================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Колатор для Whisper: паддить features та labels"""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Окремо обробляємо input features і labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Замінити padding token на -100 (ігнорується CrossEntropy)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Видалити BOS token якщо він є на початку
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class UserAudioDataset(Dataset):
    """Dataset для користувацьких записів"""

    def __init__(self, metadata_path, audio_dir, processor, max_length=480000):
        self.processor = processor
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.samples = []

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    audio_path = os.path.join(audio_dir, row['file_name'])
                    if os.path.exists(audio_path):
                        self.samples.append({
                            'audio_path': audio_path,
                            'transcription': row['transcription'],
                            'category': row.get('category', ''),
                        })

        print(f"  User dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Завантажити аудіо
        import soundfile as sf
        audio, sr = sf.read(sample['audio_path'])

        # Ресемплінг якщо потрібно
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Обрізати якщо занадто довге (30с)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]

        # Feature extraction
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]

        # Tokenize text
        labels = self.processor.tokenizer(sample['transcription']).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


def prepare_cv_dataset(processor, max_hours=5):
    """Завантажити і підготувати Common Voice Ukrainian"""
    print(f"\n  Завантаження Common Voice (uk), max {max_hours} годин...")

    try:
        cv_dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "uk",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Помилка завантаження Common Voice: {e}")
        print("  Спробуйте: huggingface-cli login")
        return None

    # Обмежити за тривалістю
    max_seconds = max_hours * 3600
    total_seconds = 0
    indices = []
    for i, sample in enumerate(cv_dataset):
        # Common Voice має поле 'audio' з 'array' і 'sampling_rate'
        duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        total_seconds += duration
        indices.append(i)
        if total_seconds >= max_seconds:
            break

    cv_dataset = cv_dataset.select(indices)
    print(f"  Common Voice: {len(cv_dataset)} samples ({total_seconds/3600:.1f} годин)")

    # Ресемплінг до 16kHz
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_cv_sample(batch):
        audio = batch["audio"]
        # Feature extraction
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=16000, return_tensors="np"
        ).input_features[0]

        # Tokenize
        labels = processor.tokenizer(batch["sentence"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    cv_dataset = cv_dataset.map(
        prepare_cv_sample,
        remove_columns=cv_dataset.column_names,
        num_proc=1,  # Windows compatibility
    )

    return cv_dataset


class CombinedDataset(Dataset):
    """Комбінований dataset: HF dataset + User dataset"""

    def __init__(self, hf_dataset, user_dataset, user_oversample=3):
        """
        user_oversample: скільки разів повторити user data (щоб збільшити його вплив)
        """
        self.hf_dataset = hf_dataset
        self.user_dataset = user_dataset
        self.user_oversample = user_oversample

        self.hf_len = len(hf_dataset) if hf_dataset else 0
        self.user_len = len(user_dataset) * user_oversample if user_dataset else 0
        self.total_len = self.hf_len + self.user_len

        print(f"  Combined: {self.hf_len} CV + {self.user_len} user = {self.total_len} total")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.hf_len:
            item = self.hf_dataset[idx]
            # HF dataset повертає numpy, конвертуємо
            return {
                "input_features": np.array(item["input_features"]),
                "labels": item["labels"],
            }
        else:
            user_idx = (idx - self.hf_len) % len(self.user_dataset)
            return self.user_dataset[user_idx]


def compute_metrics(pred, processor, metric):
    """Обчислити WER"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Замінити -100 на pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": 100 * wer}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper large-v3 with LoRA")
    parser.add_argument("--user-only", action="store_true", help="Train only on user data (no Common Voice)")
    parser.add_argument("--cv-hours", type=float, default=MAX_CV_HOURS, help=f"Max Common Voice hours (default: {MAX_CV_HOURS})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization (uses more VRAM)")
    args = parser.parse_args()

    print("=" * 60)
    print("  WHISPER LARGE-V3 FINE-TUNING (LoRA)")
    print("=" * 60)

    # Перевірити CUDA
    if not torch.cuda.is_available():
        print("CUDA не доступний! Fine-tuning потребує GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Створити директорії
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Завантажити processor
    print(f"\n  Завантаження моделі: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    # Завантажити модель
    load_kwargs = {}
    if not args.no_8bit:
        print("  8-bit квантизація увімкнена")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, **load_kwargs)

    # Налаштувати для fine-tuning
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Для gradient checkpointing

    # Підготувати для LoRA
    if not args.no_8bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA конфігурація
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        # task_type не вказуємо — PeftModelForSeq2SeqLM додає input_ids
        # який Whisper не приймає (очікує input_features)
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============ ДАНІ ============
    print("\n  Підготовка даних...")

    # Користувацькі дані
    user_dataset = None
    if os.path.exists(USER_METADATA):
        user_dataset = UserAudioDataset(USER_METADATA, USER_AUDIO_DIR, processor)
        if len(user_dataset) == 0:
            user_dataset = None
            print("  User dataset порожній, використовую лише Common Voice")

    # Common Voice
    cv_train = None
    cv_eval = None
    if not args.user_only:
        cv_train = prepare_cv_dataset(processor, max_hours=args.cv_hours)
        if cv_train:
            # Виділити 10% для evaluation
            cv_split = cv_train.train_test_split(test_size=0.1, seed=42)
            cv_train = cv_split["train"]
            cv_eval = cv_split["test"]
            print(f"  CV split: {len(cv_train)} train / {len(cv_eval)} eval")

    if user_dataset is None and cv_train is None:
        print("\nНемає даних для навчання!")
        print("Запишіть дані через collect_training_data.py або увімкніть Common Voice.")
        sys.exit(1)

    # Комбінований train dataset
    if user_dataset and cv_train:
        # User data oversampled 3x щоб мати більшу вагу
        train_dataset = CombinedDataset(cv_train, user_dataset, user_oversample=3)
    elif user_dataset:
        train_dataset = user_dataset
    else:
        train_dataset = cv_train

    # Evaluation dataset
    if user_dataset and len(user_dataset) >= 10:
        # Використати останні 10% user data як eval
        n_eval = max(1, len(user_dataset) // 10)
        # Простий split — останні n_eval зразків
        eval_indices = list(range(len(user_dataset) - n_eval, len(user_dataset)))
        eval_dataset_user = torch.utils.data.Subset(user_dataset, eval_indices)
    else:
        eval_dataset_user = None

    eval_dataset = cv_eval  # Common Voice eval як основний

    # ============ МЕТРИКА ============
    metric = evaluate.load("wer")

    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, metric)

    # ============ DATA COLLATOR ============
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ============ TRAINING ARGS ============
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=args.epochs,
        fp16=True,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=EVAL_STEPS if eval_dataset else None,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_dir=LOG_DIR,
        logging_steps=25,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,  # Windows compatibility
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ============ TRAINER ============
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn if eval_dataset else None,
        processing_class=processor.feature_extractor,
    )

    # ============ НАВЧАННЯ ============
    print("\n" + "=" * 60)
    print("  ПОЧИНАЮ НАВЧАННЯ")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {GRADIENT_ACCUMULATION} accum = {args.batch_size * GRADIENT_ACCUMULATION} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("=" * 60)

    resume_from = args.resume
    trainer.train(resume_from_checkpoint=resume_from)

    # ============ ЗБЕРЕЖЕННЯ ============
    print(f"\n  Збереження адаптера в {ADAPTER_DIR}...")
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)

    # Зберегти інфо про тренування
    train_info = {
        "base_model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGET_MODULES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "user_samples": len(user_dataset) if user_dataset else 0,
        "cv_samples": len(cv_train) if cv_train else 0,
        "gpu": gpu_name,
    }
    with open(os.path.join(ADAPTER_DIR, "training_info.json"), 'w', encoding='utf-8') as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  НАВЧАННЯ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"  Адаптер: {ADAPTER_DIR}")
    print(f"  Чекпоінти: {CHECKPOINT_DIR}")
    print(f"  Логи: {LOG_DIR}")
    print(f"\n  Запустіть evaluate_model.py для оцінки якості")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
