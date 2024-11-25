from datasets import load_dataset
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import lime
from lime.lime_text import LimeTextExplainer


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
dataset = dataset.filter(lambda ex: ex['label'] != -1)

results = []
word = "cat"
label = 2

filtered_examples = [
        ex for ex in dataset["validation"]
        if word in " ".join(ex["premise"]).lower() or word in " ".join(ex["hypothesis"]).lower()
    ]

premises = [" ".join(ex["premise"]) for ex in filtered_examples]
hypotheses = [" ".join(ex["hypothesis"]) for ex in filtered_examples]
labels = [ex["label"] for ex in filtered_examples]

output_dir = "./model"
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

inputs = tokenizer(
        premises, hypotheses,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

with torch.no_grad():
        logits = model(**inputs).logits
        predictions = logits.argmax(dim=-1).cpu().numpy()
    
# Calculate metrics
correct_label = accuracy_score(labels, predictions) * 100
misclassified_examples = [i for i, pred in enumerate(predictions) if pred != labels[i]]
label_misclassified = sum(predictions[i] == label for i in misclassified_examples) / len(misclassified_examples) * 100 if misclassified_examples else 0

def predict_proba(texts):
    """
    Predict probabilities for entailment, neutral, and contradiction.
    LIME expects probabilities as output.
    """
    # Tokenize and predict using the model
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs).logits.softmax(dim=-1)  # Convert logits to probabilities
    
    return outputs.cpu().numpy()  # Convert to numpy array

explainer = LimeTextExplainer(class_names=["entailment", "neutral", "contradiction"])

lime_scores = []
for ex in filtered_examples:
    combined_text = f"{ex['premise']} {ex['hypothesis']}"
    
    explanation = explainer.explain_instance(
        combined_text,
        predict_proba,
        num_features=10,
        labels=[0, 1, 2]
    )
    
    for word_score in explanation.as_list(label=label):
                if word_score[0] == word:
                    lime_scores.append(word_score[1])
                    
    print(sum(lime_scores) / len(lime_scores) if lime_scores else 0)
