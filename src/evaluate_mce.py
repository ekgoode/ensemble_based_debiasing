import os
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter, defaultdict
import torch.optim as optim
from tqdm import tqdm
from train_mce import HighCapacityModel, MixedCapacityEnsemble, SNLIDataset, build_bow_vocab, prepare_snli_dataloaders, train_mixed_capacity_ensemble, evaluate_mixed_capacity_ensemble, train

checkpoint_path = "./mce_results/archivev2/mixed_capacity_ensemble_epoch_3.pt"
optimizer_path = "./mce_results/archivev2/optimizer_epoch_3.pt"

electra_model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator", num_labels=3
)
vocab_size = 1028
bow_model = nn.Sequential(
    nn.Linear(vocab_size * 2, 300),
    nn.ReLU(),
    nn.Linear(300, 3)
)

model = MixedCapacityEnsemble(electra_model, bow_model, num_classes=3)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

optimizer = optim.Adam(model.parameters(), lr=5e-5)
optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device('cpu')))

dataset = load_dataset("snli")
train_dataset = dataset["train"].filter(lambda example: example["label"] != -1)

def build_bow_vocab(dataset, vocab_size=10000):
    counter = Counter()
    for example in dataset:
        counter.update(example["premise"].lower().split())
        counter.update(example["hypothesis"].lower().split())
    most_common = counter.most_common(vocab_size)
    return {word: idx for idx, (word, _) in enumerate(most_common)}

bow_vocab = build_bow_vocab(train_dataset, vocab_size=1028)

output_dir = "./model"

tokenizer = AutoTokenizer.from_pretrained(output_dir)

filtered_vocab = word_counts_df[word_counts_df['count'] >= 20]['word'].tolist()

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
num_labels = len(label_map)

results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Starting z-score computation...")
for counter, word in enumerate(tqdm(filtered_vocab, desc="Processing words")):
    electra_input = tokenizer(word, "", return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    electra_input = {k: v.to(device) for k, v in electra_input.items()}

    bow_input = torch.zeros(len(bow_vocab) * 2, dtype=torch.float32).to(device)
    if word in bow_vocab:
        bow_input[bow_vocab[word]] = 1.0

    with torch.no_grad():
        ensemble_logits, _ = model(electra_input, bow_input.unsqueeze(0))
        probabilities = F.softmax(ensemble_logits, dim=-1).squeeze(0)

    for label_idx in range(num_labels):
        results.append({
            "word": word,
            "label": label_map[label_idx],
            "p(y|x_i)": probabilities[label_idx].item()
        })

results_df = pd.DataFrame(results)

results_df = results_df.merge(word_counts_df[['word', 'count']], on='word', how='left')

results_df['z*'] = (results_df['p(y|x_i)'] - (1 / num_labels)) / (
    ((1 / num_labels) * (1 - (1 / num_labels)) / results_df['count']) ** 0.5
)


z_star_df = results_df.pivot(index="word", columns="label", values="z*")
strongest_classes = z_star_df.idxmax(axis=1)
results_df['strongest_class'] = results_df['word'].map(strongest_classes)
final_results = []

for label in label_map.values():
    class_words = results_df[results_df['strongest_class'] == label]

    high_group = class_words.nlargest(20, "z*")
    low_group = class_words.nsmallest(20, "z*")
    
    delta_p_y = (high_group['p(y|x_i)'].sum() - low_group['p(y|x_i)'].sum())

    final_results.append({
        "Dataset": "SNLI",
        "Class": label,
        "Δp_y": f"{delta_p_y:.1f} %"
    })

delta_p_y_df = pd.DataFrame(final_results)

print(delta_p_y_df)

top_words_by_class = {}
for label in label_map.values():
    class_words = results_df[results_df['label'] == label]
    top_words = class_words.nlargest(20, "z*")["word"].tolist()
    top_words_by_class[label] = set(top_words)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
_, _, test_loader, bow_vocab = prepare_snli_dataloaders(batch_size=1)
results_by_class = []

for label, top_words in top_words_by_class.items():
    total_correct = 0
    total_examples = 0
    total_miss = 0

    for batch in tqdm(test_loader, desc=f"Processing {label} examples"):
        electra_input = batch["electra_input"]
        bow_input = batch["bow_input"]
        label_id = batch["label"].item()

        tokens = tokenizer.batch_decode(electra_input["input_ids"], skip_special_tokens=True)[0].split()

        if not any(word in tokens for word in top_words):
            continue

        total_examples += 1

        electra_input = {k: v.to(device) for k, v in electra_input.items()}
        bow_input = bow_input.to(device)

        with torch.no_grad():
            ensemble_logits, _ = model(electra_input, bow_input)
            probabilities = F.softmax(ensemble_logits, dim=-1)

        predicted_label = torch.argmax(probabilities, dim=-1).item()
        true_logit = probabilities[0, label_id].item()

        total_correct += int(predicted_label == label_id)
        total_miss += (1 - true_logit)

    accuracy = total_correct / total_examples if total_examples > 0 else 0
    avg_miss = total_miss / total_examples if total_examples > 0 else 0

    results_by_class.append({
        "Class": label,
        "Accuracy": f"{accuracy:.2%}",
        "Average Miss": f"{avg_miss:.4f}",
        "Examples Analyzed": total_examples,
    })

final_table = pd.DataFrame(results_by_class)
print(final_table)

datasets = {
    "SNLI": snli_loader,
    "HANS": hans_loader,
    "HANSTRAIN": hans_train
}

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_dataloader(dataset_name, split, bow_vocab, electra_tokenizer, batch_size=32, max_seq_len=128):
    dataset = load_dataset(dataset_name, split=split)
    
    if "label" in dataset.features and -1 in dataset["label"]:
        dataset = dataset.filter(lambda example: example["label"] != -1)
    
    dataset = SNLIDataset(dataset, electra_tokenizer, bow_vocab, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

electra_tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

snli_loader = prepare_dataloader("snli", "test", bow_vocab, electra_tokenizer)
hans_loader = prepare_dataloader("hans", "validation", bow_vocab, electra_tokenizer)
hans_train = prepare_dataloader("hans", "train", bow_vocab, electra_tokenizer)

def evaluate_mixed_capacity_ensemble(model, dataloader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            electra_input = {k: v.to(device) for k, v in batch["electra_input"].items()}
            bow_input = batch["bow_input"].to(device)
            labels = batch["label"].to(device)

            ensemble_logits = model(electra_input, bow_input)

            if isinstance(ensemble_logits, tuple):
                ensemble_logits = ensemble_logits[0]

            loss = criterion(ensemble_logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(ensemble_logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total



for name, loader in datasets.items():
    loss, accuracy = evaluate_mixed_capacity_ensemble(model, loader, criterion, device=device)
    print(f"{name} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


