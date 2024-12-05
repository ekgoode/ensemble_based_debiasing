import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import norm
from train_mce import MixedCapacityEnsemble, prepare_snli_dataloaders
from collections import Counter, defaultdict

CHECKPOINT_PATH = "./models/mixed_capacity_ensemble_epoch_3.pt"
OPTIMIZER_PATH = "./models/optimizer_epoch_3.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

def clean_text(text: str):
    """
    Clean and tokenize text by removing punctuation and converting to lowercase.
    """
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def preprocess_dataset(dataset):
    """
    Preprocess the SNLI dataset: clean text and combine premise and hypothesis.
    """
    def clean_example(example):
        example["premise"] = clean_text(example["premise"])
        example["hypothesis"] = clean_text(example["hypothesis"])
        example["combined"] = example["premise"] + example["hypothesis"]
        return example

    return dataset.map(clean_example)

def compute_word_statistics(dataset):
    """
    Compute word-level statistics (count, label distribution, z-scores).
    """
    all_tokens = [token for example in dataset["train"]["combined"] for token in example]
    word_counts = Counter(all_tokens)

    label_word_counts = defaultdict(lambda: defaultdict(int))
    for example in dataset["train"]:
        label = LABEL_MAP[example["label"]]
        for word in example["combined"]:
            label_word_counts[label][word] += 1

    # Merge counts into DataFrame
    word_counts_df = pd.DataFrame.from_dict(word_counts, orient="index", columns=["count"]).reset_index()
    word_counts_df.rename(columns={"index": "word"}, inplace=True)

    for label, counts in label_word_counts.items():
        label_df = pd.DataFrame.from_dict(counts, orient="index", columns=[f"{label}_count"]).reset_index()
        label_df.rename(columns={"index": "word"}, inplace=True)
        word_counts_df = word_counts_df.merge(label_df, on="word", how="left").fillna(0)

    word_counts_df = word_counts_df[word_counts_df["count"] >= 3].copy()
    word_counts_df["critical_z"] = norm.ppf(1 - (0.01 / word_counts_df.shape[0]))
    return word_counts_df

def compute_z_scores(word_counts_df):
    """
    Compute z-scores for word-label probabilities.
    """
    for label in LABEL_MAP.values():
        p_hat = f"p_hat_{label}"
        z_stat = f"z_stat_{label}"
        label_count = f"{label}_count"

        word_counts_df[p_hat] = word_counts_df[label_count] / word_counts_df["count"]
        word_counts_df[z_stat] = (word_counts_df[p_hat] - (1 / 3)) / (
            ((1 / 3) * (1 - (1 / 3)) / word_counts_df["count"]) ** 0.5
        )
    return word_counts_df

def generate_table_4(word_counts_df, model, tokenizer, bow_vocab):
    """
    Generate results for Table 4: Compute z-scores using the MCE model.
    """
    results = []
    model.to(DEVICE)

    for word in tqdm(word_counts_df["word"], desc="Processing words for Table 4"):
        electra_input = tokenizer(word, "", return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        electra_input = {k: v.to(DEVICE) for k, v in electra_input.items()}

        bow_input = torch.zeros(len(bow_vocab) * 2, dtype=torch.float32).to(DEVICE)
        if word in bow_vocab:
            bow_input[bow_vocab[word]] = 1.0

        with torch.no_grad():
            ensemble_logits, _ = model(electra_input, bow_input.unsqueeze(0))
            probabilities = F.softmax(ensemble_logits, dim=-1).squeeze(0)

        for label_idx, label in LABEL_MAP.items():
            results.append({"word": word, "label": label, "p(y|x_i)": probabilities[label_idx].item()})

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(word_counts_df[["word", "count"]], on="word", how="left")
    results_df["z*"] = (results_df["p(y|x_i)"] - (1 / len(LABEL_MAP))) / (
        ((1 / len(LABEL_MAP)) * (1 - (1 / len(LABEL_MAP))) / results_df["count"]) ** 0.5
    )
    return results_df

def generate_table_5(results_df):
    """
    Generate results for Table 5: Compare z-scores for high and low groups.
    """
    final_results = []
    for label in LABEL_MAP.values():
        class_words = results_df[results_df["label"] == label]
        high_group = class_words.nlargest(20, "z*")
        low_group = class_words.nsmallest(20, "z*")

        delta_p_y = (high_group["p(y|x_i)"].sum() - low_group["p(y|x_i)"].sum())
        final_results.append({"Dataset": "SNLI", "Class": label, "Δp_y": f"{delta_p_y:.1f} %"})
    return pd.DataFrame(final_results)

def evaluate_on_datasets(model, datasets, criterion):
    """
    Evaluate the MCE model on SNLI and HANS datasets.
    """
    results = []
    for name, loader in datasets.items():
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(loader, desc=f"Evaluating {name}"):
            electra_input = {k: v.to(DEVICE) for k, v in batch["electra_input"].items()}
            bow_input = batch["bow_input"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                ensemble_logits, _ = model(electra_input, bow_input)
                loss = criterion(ensemble_logits, labels)
                total_loss += loss.item()

                preds = ensemble_logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        results.append({"Dataset": name, "Loss": total_loss / len(loader), "Accuracy": accuracy})
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Preprocess SNLI dataset
    dataset = preprocess_dataset(load_dataset("snli"))
    word_counts_df = compute_word_statistics(dataset)
    word_counts_df = compute_z_scores(word_counts_df)

    # Load MCE model
    electra_model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
    bow_model = nn.Sequential(nn.Linear(1028 * 2, 300), nn.ReLU(), nn.Linear(300, 3))
    model = MixedCapacityEnsemble(electra_model, bow_model, num_classes=3)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))

    # Generate Tables
    table_4_results = generate_table_4(word_counts_df, model, AutoTokenizer.from_pretrained("google/electra-small-discriminator"), bow_vocab={})
    table_5_results = generate_table_5(table_4_results)
    print(table_5_results)

    # Evaluate on SNLI and HANS
    datasets = {
        "SNLI": prepare_snli_dataloaders("snli", "test", batch_size=32)[2],
        "HANS": prepare_snli_dataloaders("hans", "validation", batch_size=32)[2],
    }
    evaluation_results = evaluate_on_datasets(model, datasets, criterion=torch.nn.CrossEntropyLoss())
    print(evaluation_results)
