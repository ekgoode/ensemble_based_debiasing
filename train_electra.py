import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

def initialize_electra(model_name="google/electra-small-discriminator", num_labels=3):
    """
    Initialize the ELECTRA-small model and tokenizer for sequence classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def preprocess_dataset(dataset, tokenizer, max_seq_len=128):
    """
    Tokenize the dataset using the ELECTRA tokenizer.
    """
    def preprocess(example):
        return tokenizer(
            example['premise'], example['hypothesis'],
            truncation=True, padding='max_length', max_length=max_seq_len
        )

    dataset = dataset.filter(lambda ex: ex['label'] != -1)  # Remove invalid labels
    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")  # Ensure labels are correctly named
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def compute_metrics(pred):
    """
    Compute accuracy for the predictions.
    """
    predictions = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def train_electra(model, tokenizer, dataset, output_dir="./electra-snli-model", epochs=3, batch_size=16):
    """
    Train the ELECTRA-small model on the SNLI dataset.
    """
    encoded_dataset = preprocess_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer

def evaluate_model(trainer, dataset, tokenizer, split="test"):
    """
    Evaluate the trained model on a specified dataset split.
    """
    dataset = preprocess_dataset(dataset, tokenizer)
    results = trainer.evaluate(eval_dataset=dataset[split])
    print(f"Validation Accuracy: {results['eval_accuracy']:.4f}")
    return results

if __name__ == "__main__":
    model_name = "google/electra-small-discriminator"
    output_dir = "./electra-snli-model"

    # Load Dataset
    dataset = load_dataset("snli")
    tokenizer, model = initialize_electra(model_name)

    # Train Model
    print("Training ELECTRA-small model...")
    trainer = train_electra(model, tokenizer, dataset, output_dir=output_dir)

    # Evaluate Model
    print("Evaluating on SNLI test set...")
    evaluate_model(trainer, dataset, tokenizer, split="test")
