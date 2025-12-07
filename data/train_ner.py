import json
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# --- KONFIGURACJA ---
INPUT_FILE = "data_ner_ready.jsonl"
MODEL_CHECKPOINT = "allegro/herbert-base-cased"  # Najlepszy standardowy model dla PL
OUTPUT_DIR = "./final_ner_model"
EPOCHS = 4
BATCH_SIZE = 8  # Zmniejsz do 4, jeśli masz mało RAM/VRAM

# 1. Wczytanie danych
print(f"Wczytywanie danych z {INPUT_FILE}...")
data = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Konwersja do formatu HuggingFace Dataset
raw_dataset = Dataset.from_list(data)
# Podział na zbiór treningowy i testowy (10% na testy)
dataset = raw_dataset.train_test_split(test_size=0.1)

# 2. Tworzenie mapy etykiet (Label Map)
# Musimy znaleźć wszystkie unikalne tagi w całym zbiorze
unique_tags = set()
for item in data:
    unique_tags.update(item['ner_tags'])

label_list = sorted(list(unique_tags))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Liczba unikalnych tagów: {len(label_list)}")
print(f"Przykładowe tagi: {label_list[:10]}")

# 3. Tokenizacja i wyrównanie etykiet
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Tokeny specjalne (np. [CLS], [SEP]) dostają -100 (ignorowane przez model)
            if word_idx is None:
                label_ids.append(-100)
            # Jeśli to nowy wyraz, bierzemy jego etykietę
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # Jeśli to kolejna część tego samego wyrazu (sub-token), też dajemy -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 4. Metryki (SeqEval)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 6. Trening
args = TrainingArguments(
    OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2, # Trzymaj tylko 2 ostatnie checkpointy
    load_best_model_at_end=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n--- ROZPOCZYNANIE TRENINGU ---\n")
trainer.train()

# 7. Zapis modelu końcowego
print(f"Zapisywanie modelu do {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Gotowe!")