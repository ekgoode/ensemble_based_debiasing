import numpy as np
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

def initialize_tokenizer(model_name="google/electra-small-discriminator"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def build_bow_vocab(dataset, vocab_size=10000):
    """
    Build a Bag-of-Words vocabulary based on word frequency in the dataset.
    """
    counter = Counter()
    for example in dataset:
        counter.update(example["premise"].lower().split())
        counter.update(example["hypothesis"].lower().split())
    most_common = counter.most_common(vocab_size)
    return {word: idx for idx, (word, _) in enumerate(most_common)}

def preprocess_dataset(dataset, tokenizer, max_seq_len=128):
    """
    Tokenize the dataset using the specified tokenizer.
    """
    def preprocess(example):
        return tokenizer(
            example['premise'], example['hypothesis'],
            truncation=True, padding='max_length', max_length=max_seq_len
        )
    return dataset.map(preprocess, batched=True)

def filter_invalid_labels(dataset):
    """
    Remove examples with invalid labels (label == -1) from the dataset.
    """
    return dataset.filter(lambda example: example["label"] != -1)

def prepare_dataloader(dataset_name, split, tokenizer, bow_vocab, batch_size=32, max_seq_len=128):
    """
    Prepare a PyTorch DataLoader for a given dataset and split.
    """
    dataset = load_dataset(dataset_name, split=split)
    dataset = filter_invalid_labels(dataset)
    dataset = preprocess_dataset(dataset, tokenizer, max_seq_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    tokenizer = initialize_tokenizer()
    dataset = load_dataset("snli")["train"]
    dataset = filter_invalid_labels(dataset)
    bow_vocab = build_bow_vocab(dataset)
    print(f"Vocabulary size: {len(bow_vocab)}")
