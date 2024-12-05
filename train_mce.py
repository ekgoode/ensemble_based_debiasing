import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from collections import Counter

def initialize_tokenizer(model_name="google/electra-small-discriminator"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

class MixedCapacityEnsemble(nn.Module):
    def __init__(self, high_capacity_model, low_capacity_vocab_size, num_labels):
        """
        Mixed Capacity Ensemble combines a high-capacity ELECTRA model
        with a low-capacity Bag-of-Words (BoW) model.
        """
        super(MixedCapacityEnsemble, self).__init__()
        self.high_capacity_model = high_capacity_model
        self.low_capacity_model = nn.Sequential(
            nn.Linear(low_capacity_vocab_size, 300),
            nn.ReLU(),
            nn.Linear(300, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, high_inputs, low_inputs):
        """
        Forward pass through both models.
        """
        # High-capacity model
        high_output = self.high_capacity_model(**high_inputs).logits

        # Low-capacity model
        low_output = self.low_capacity_model(low_inputs)

        # Combine predictions
        combined_logits = high_output + low_output
        return combined_logits

def build_bow_vocab(dataset, vocab_size=10000):
    counter = Counter()
    for example in dataset:
        counter.update(example["premise"].lower().split())
        counter.update(example["hypothesis"].lower().split())
    most_common = counter.most_common(vocab_size)
    return {word: idx for idx, (word, _) in enumerate(most_common)}

def prepare_bow_features(dataset, bow_vocab):
    """
    Convert dataset examples into Bag-of-Words feature matrices.
    """
    def to_bow(example):
        bow_vector = torch.zeros(len(bow_vocab))
        for word in example["premise"].lower().split() + example["hypothesis"].lower().split():
            if word in bow_vocab:
                bow_vector[bow_vocab[word]] += 1
        return {"bow_features": bow_vector}

    return dataset.map(to_bow)

def preprocess_dataset(dataset, tokenizer, bow_vocab, max_seq_len=128):
    """
    Tokenize dataset for high-capacity model and prepare BoW features for low-capacity model.
    """
    def tokenize_and_bow(example):
        tokenized = tokenizer(
            example["premise"], example["hypothesis"],
            truncation=True, padding="max_length", max_length=max_seq_len
        )
        bow_features = torch.zeros(len(bow_vocab))
        for word in example["premise"].lower().split() + example["hypothesis"].lower().split():
            if word in bow_vocab:
                bow_features[bow_vocab[word]] += 1
        return {**tokenized, "bow_features": bow_features}

    dataset = dataset.filter(lambda ex: ex['label'] != -1)  # Remove invalid labels
    dataset = dataset.map(tokenize_and_bow)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "bow_features", "label"])
    return dataset

def train_mce(high_capacity_model, bow_vocab, dataset, output_dir="./mce-snli-model", epochs=3, batch_size=16):
    """
    Train the Mixed Capacity Ensemble model.
    """
    mce_model = MixedCapacityEnsemble(high_capacity_model, len(bow_vocab), num_labels=3)

    # Preprocess Dataset
    tokenizer = initialize_tokenizer()
    processed_dataset = preprocess_dataset(dataset, tokenizer, bow_vocab)

    # DataLoader
    train_dataset = processed_dataset["train"]
    val_dataset = processed_dataset["validation"]

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

    def compute_metrics(pred):
        predictions = pred.predictions.argmax(axis=-1)
        labels = pred.label_ids
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    trainer = Trainer(
        model=mce_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer

if __name__ == "__main__":
    
    #Load dataset
    dataset = load_dataset("snli")

    # Initialize model objects
    bow_vocab = build_bow_vocab(dataset["train"])
    high_capacity_model = AutoModelForSequenceClassification.from_pretrained(
        "google/electra-small-discriminator", num_labels=3
    )

    # Train model
    print("Training MCE...")
    train_mce(high_capacity_model, bow_vocab, dataset)
