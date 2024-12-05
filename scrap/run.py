# This will eventually turn into the run script
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# Load dataset
dataset = load_dataset("snli")

# Initialize model + tokenizer
model_name = "google/electra-small-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def preprocess(example):
    return tokenizer(example['premise'], example['hypothesis'], truncation=True, padding='max_length', max_length=128)

encoded_snli = snli.map(preprocess, batched=True)
encoded_snli = encoded_snli.rename_column("label", "labels")  # Ensure labels are named correctly
encoded_snli.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_snli['train'],
    eval_dataset=encoded_snli['validation'],
)
trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")