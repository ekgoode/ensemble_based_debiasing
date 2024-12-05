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
from collections import Counter
import torch.optim as optim
from tqdm import tqdm
import string
from collections import Counter, defaultdict
from train_mce import  SNLIDataset, prepare_snli_dataloaders

output_dir = "./model"

model = AutoModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

dataset = load_dataset("snli")

def clean_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text.split()

def clean_snli(example):
    example["premise"] = clean_text(example["premise"])
    example["hypothesis"] = clean_text(example["hypothesis"])
    example["combined"] = example["premise"] + example["hypothesis"]
    return example

dataset = dataset.map(clean_snli)

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

def compute_metrics(pred):
    predictions = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics,
)

validation_results = trainer.evaluate()
print(f"Validation accuracy: {validation_results['eval_accuracy']:.4f}")
all_tokens = [token for example in dataset['train']['combined'] for token in example]

word_counts = Counter(all_tokens)
word_counts_df = pd.DataFrame(word_counts.items(), columns=["word", "count"])
word_counts_df = word_counts_df.sort_values(by="count", ascending=False).reset_index(drop=True)
print(word_counts_df.head())
label_word_counts = {
    "entailment": defaultdict(int),
    "contradiction": defaultdict(int),
    "neutral": defaultdict(int),
}

for example in dataset["train"]:
    label = example["label"]
    label_name = ["entailment", "neutral", "contradiction"][label]
    for word in example["combined"]:
        label_word_counts[label_name][word] += 1

entailment_counts = pd.DataFrame(list(label_word_counts["entailment"].items()), columns=["word", "entailment_count"])
contradiction_counts = pd.DataFrame(list(label_word_counts["contradiction"].items()), columns=["word", "contradiction_count"])
neutral_counts = pd.DataFrame(list(label_word_counts["neutral"].items()), columns=["word", "neutral_count"])

word_counts_df = word_counts_df.merge(entailment_counts, on="word", how="left").fillna(0)
word_counts_df = word_counts_df.merge(contradiction_counts, on="word", how="left").fillna(0)
word_counts_df = word_counts_df.merge(neutral_counts, on="word", how="left").fillna(0)

word_counts_df[["entailment_count", "contradiction_count", "neutral_count"]] = word_counts_df[
    ["entailment_count", "contradiction_count", "neutral_count"]
].astype(int)

word_counts_df = word_counts_df[word_counts_df['count'] >= 3].copy()
alpha = 0.01/word_counts_df.shape[0]
z_star = norm.ppf(1 - alpha)
print(z_star)
print(word_counts_df.shape[0])
word_counts_df['alpha'] = 0.01
word_counts_df['p_hat_entailment'] = word_counts_df['entailment_count']/word_counts_df['count']
word_counts_df['p_hat_contradiction'] = word_counts_df['contradiction_count']/word_counts_df['count']
word_counts_df['p_hat_neutral'] = word_counts_df['neutral_count']/word_counts_df['count']
word_counts_df['z_stat_entailment'] = (word_counts_df['p_hat_entailment']-(1/3)) / np.sqrt(((1/3)*(1-(1/3)))/word_counts_df['count'])
word_counts_df['z_stat_contradiction'] = (word_counts_df['p_hat_contradiction']-(1/3)) / np.sqrt(((1/3)*(1-(1/3)))/word_counts_df['count'])
word_counts_df['z_stat_neutral'] = (word_counts_df['p_hat_neutral']-(1/3)) / np.sqrt(((1/3)*(1-(1/3)))/word_counts_df['count'])
word_counts_df['critical_z'] = z_star
word_counts_df.head()
filtered_vocab = word_counts_df[word_counts_df['count'] >= 20]['word'].tolist()

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
num_labels = len(label_map)

results = []
counter = 0
for word in filtered_vocab:
    if counter % 1000 == 0:
        print(f"Processing word {counter}: {word}")
    counter += 1
    premise_input = tokenizer(
        word, "", return_tensors="pt", truncation=True, padding="max_length", max_length=tokenizer.model_max_length
    )
    hypothesis_input = tokenizer(
        "", word, return_tensors="pt", truncation=True, padding="max_length", max_length=tokenizer.model_max_length
    )
    
    with torch.no_grad():
        premise_output = model(**premise_input).logits.softmax(dim=-1)
        hypothesis_output = model(**hypothesis_input).logits.softmax(dim=-1)
    
    avg_probabilities = (premise_output + hypothesis_output) / 2

    for label_idx in range(num_labels):
        results.append({
            "word": word,
            "label": label_map[label_idx],
            "p(y|x_i)": avg_probabilities[0, label_idx].item()
        })

results_df = pd.DataFrame(results)

results_df = results_df.merge(
    word_counts_df[['word', 'entailment_count', 'contradiction_count', 'neutral_count']],
    on='word',
    how='left'
)

def get_class_count(row):
    if row['label'] == "entailment":
        return row['entailment_count']
    elif row['label'] == "contradiction":
        return row['contradiction_count']
    elif row['label'] == "neutral":
        return row['neutral_count']
    return 0

results_df['class_count'] = results_df.apply(get_class_count, axis=1)

results_df['class_count'] = results_df['class_count'].fillna(1)

results_df['z*'] = (results_df['p(y|x_i)'] - (1/3)) / np.sqrt((1/3) * (1 - (1/3)) / results_df['class_count'])

z_star_df = results_df.pivot(index="word", columns="label", values="z*")
strongest_classes = z_star_df.idxmax(axis=1)
results_df['strongest_class'] = results_df['word'].map(strongest_classes)
final_results = []

for label in label_map.values():
    class_words = results_df[results_df['strongest_class'] == label]
    
    high_group = class_words.nlargest(50, "z*")
    low_group = class_words.nsmallest(50, "z*")
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
