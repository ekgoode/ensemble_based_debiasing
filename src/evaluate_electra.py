import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from train_mce import prepare_snli_dataloaders
import string

OUTPUT_DIR = "./models"
SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Load datasets
snli_dataset = load_dataset("snli")
hans_dataset = load_dataset("hans", split="validation")

def clean_text(text: str):
    """
    Clean and tokenize text by removing punctuation and converting to lowercase.
    """
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def preprocess_dataset(dataset, tokenizer):
    """
    Preprocess the dataset: clean text and combine premise and hypothesis.
    """
    dataset = dataset.map(lambda example: {
        "premise": clean_text(example["premise"]),
        "hypothesis": clean_text(example["hypothesis"]),
        "combined": clean_text(example["premise"]) + clean_text(example["hypothesis"])
    })
    dataset = dataset.map(
        lambda example: tokenizer(example["premise"], example["hypothesis"], 
                                  truncation=True, padding="max_length", max_length=tokenizer.model_max_length),
        batched=True
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

# Preprocess datasets
snli_dataset = preprocess_dataset(snli_dataset, tokenizer)
hans_dataset = preprocess_dataset(hans_dataset, tokenizer)

def evaluate_dataset_accuracy(model, dataset, dataset_name):
    """
    Evaluate accuracy of the model on a given dataset.
    """
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=lambda pred: {"accuracy": (pred.predictions.argmax(axis=-1) == pred.label_ids).mean()}
    )
    results = trainer.evaluate()
    print(f"{dataset_name} Validation Accuracy: {results['eval_accuracy']:.4f}")
    return results['eval_accuracy']

def generate_table_1(dataset):
    """
    Generate Table 1: Word-level statistics for the dataset.
    """
    all_tokens = [token for example in dataset['train']['combined'] for token in example]
    word_counts = Counter(all_tokens)

    label_word_counts = {label: defaultdict(int) for label in ["entailment", "neutral", "contradiction"]}
    for example in dataset["train"]:
        label = ["entailment", "neutral", "contradiction"][example["labels"]]
        for word in example["combined"]:
            label_word_counts[label][word] += 1

    word_counts_df = pd.DataFrame.from_dict(word_counts, orient="index", columns=["count"]).reset_index()
    word_counts_df = word_counts_df.rename(columns={"index": "word"})

    for label in label_word_counts.keys():
        label_counts = pd.DataFrame.from_dict(label_word_counts[label], orient="index", columns=[f"{label}_count"]).reset_index()
        word_counts_df = word_counts_df.merge(label_counts, on="word", how="left").fillna(0)

    word_counts_df["critical_z"] = norm.ppf(1 - (0.01 / word_counts_df.shape[0]))
    word_counts_df["p_hat_entailment"] = word_counts_df["entailment_count"] / word_counts_df["count"]
    word_counts_df["p_hat_contradiction"] = word_counts_df["contradiction_count"] / word_counts_df["count"]
    word_counts_df["p_hat_neutral"] = word_counts_df["neutral_count"] / word_counts_df["count"]

    print(word_counts_df.head())
    return word_counts_df

def generate_table_2(model, tokenizer, dataset, top_words_by_class):
    """
    Generate Table 2: Evaluate model predictions on examples with top artifact words.
    """
    _, _, test_loader, _ = prepare_snli_dataloaders(batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results_by_class = []
    for label, top_words in top_words_by_class.items():
        total_correct, total_examples, total_miss = 0, 0, 0

        for batch in tqdm(test_loader, desc=f"Processing {label} examples"):
            electra_input = batch["electra_input"]
            tokens = tokenizer.batch_decode(electra_input["input_ids"], skip_special_tokens=True)[0].split()

            if not any(word in tokens for word in top_words):
                continue

            total_examples += 1
            electra_input = {k: v.to(device) for k, v in electra_input.items()}

            with torch.no_grad():
                logits = model(**electra_input).logits
                probabilities = F.softmax(logits, dim=-1)

            predicted_label = probabilities.argmax(dim=-1).item()
            true_label = batch["label"].item()
            true_logit = probabilities[0, true_label].item()

            total_correct += int(predicted_label == true_label)
            total_miss += (1 - true_logit)

        accuracy = total_correct / total_examples if total_examples > 0 else 0
        avg_miss = total_miss / total_examples if total_examples > 0 else 0

        results_by_class.append({
            "Class": label,
            "Accuracy": f"{accuracy:.2%}",
            "Average Miss": f"{avg_miss:.4f}",
            "Examples Analyzed": total_examples,
        })

    return pd.DataFrame(results_by_class)

if __name__ == "__main__":
    # Accuracy on SNLI
    print("Evaluating SNLI...")
    evaluate_dataset_accuracy(model, snli_dataset["test"], "SNLI")

    # Accuracy on HANS
    print("Evaluating HANS...")
    evaluate_dataset_accuracy(model, hans_dataset, "HANS")

    # Table 1 results
    word_counts_df = generate_table_1(snli_dataset)

    # Table 2 results
    top_words_by_class = {
        "entailment": set(word_counts_df.nlargest(20, "p_hat_entailment")["word"]),
        "neutral": set(word_counts_df.nlargest(20, "p_hat_neutral")["word"]),
        "contradiction": set(word_counts_df.nlargest(20, "p_hat_contradiction")["word"]),
    }
    table_2_results = generate_table_2(model, tokenizer, snli_dataset, top_words_by_class)
    print(table_2_results)
