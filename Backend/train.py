from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import torch
import evaluate
from collections import Counter

# Load your labeled folders: Data/dry, Data/damp, Data/wet
dataset = load_dataset("imagefolder", data_dir="../Data")

# Split into train/test (80/20)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

model_name = "google/mobilenet_v2_1.0_224"  # fast to fine-tune, good for hackathon time budget
processor = AutoImageProcessor.from_pretrained(model_name)

def transform(example_batch):
    inputs = processor([img.convert("RGB") for img in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

labels = dataset["train"].features["label"].names  # grab label names BEFORE with_transform

dataset = dataset.with_transform(transform)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
    ignore_mismatched_sizes=True
)

# ---- Metrics: overall accuracy + per-class / macro precision, recall, f1 ----
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    refs = eval_pred.label_ids

    results = {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=refs)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=refs, average="macro")["f1"],
        "precision_macro": precision_metric.compute(predictions=predictions, references=refs, average="macro")["precision"],
        "recall_macro": recall_metric.compute(predictions=predictions, references=refs, average="macro")["recall"],
    }

    # Per-class breakdown so you can see if any single class (e.g. damp) is underperforming
    per_class_f1 = f1_metric.compute(predictions=predictions, references=refs, average=None)["f1"]
    per_class_recall = recall_metric.compute(predictions=predictions, references=refs, average=None)["recall"]
    per_class_precision = precision_metric.compute(predictions=predictions, references=refs, average=None)["precision"]

    for i, label_name in enumerate(labels):
        results[f"f1_{label_name}"] = per_class_f1[i]
        results[f"recall_{label_name}"] = per_class_recall[i]
        results[f"precision_{label_name}"] = per_class_precision[i]

    return results

# ---- Class weighting to offset dataset imbalance (e.g. fewer 'damp' samples) ----
train_labels = dataset["train"]["label"]
label_counts = Counter(train_labels)
total_samples = sum(label_counts.values())
num_classes = len(labels)

class_weights = torch.tensor(
    [total_samples / (num_classes * label_counts[i]) for i in range(num_classes)],
    dtype=torch.float
)

print("Class counts:", {labels[i]: label_counts[i] for i in range(num_classes)})
print("Class weights:", {labels[i]: round(class_weights[i].item(), 3) for i in range(num_classes)})

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_batch = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, num_classes), labels_batch.view(-1))
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # macro f1 is a fairer target than accuracy given class imbalance
    remove_unused_columns=False,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("model")
processor.save_pretrained("model")

# Final eval summary printed to console for a quick sanity check
final_metrics = trainer.evaluate()
print("\nFinal evaluation metrics:")
for k, v in final_metrics.items():
    print(f"  {k}: {v}")

print("\nTraining complete. Model saved to Backend/model/")