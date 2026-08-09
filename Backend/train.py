from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

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

dataset = dataset.with_transform(transform)
labels = dataset["train"].features["label"].names

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
    ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

args = TrainingArguments(
    output_dir="model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("model")
processor.save_pretrained("model")

print("Training complete. Model saved to Backend/model/")