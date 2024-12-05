from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
from collections import Counter


class SNLIDataset(Dataset):
    def __init__(self, dataset, electra_tokenizer, bow_vocab, max_seq_len=128):
        self.dataset = dataset
        self.electra_tokenizer = electra_tokenizer
        self.bow_vocab = bow_vocab
        self.max_seq_len = max_seq_len
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def bow_tokenize(self, sentence):
        tokens = sentence.lower().split()
        bow_vector = np.zeros(len(self.bow_vocab), dtype=np.float32)
        for token in tokens:
            if token in self.bow_vocab:
                bow_vector[self.bow_vocab[token]] += 1
        return bow_vector

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        # Handle integer and string labels
        label = example["label"]
        if isinstance(label, str):
            label = self.label_map[label]
        
        # Tokenize for ELECTRA
        electra_inputs = self.electra_tokenizer(
            premise, hypothesis,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            padding="max_length"
        )

        # Remove unnecessary batch dimension
        electra_inputs = {k: v.squeeze(0) for k, v in electra_inputs.items()}

        # Tokenize for BoW
        bow_premise = self.bow_tokenize(premise)
        bow_hypothesis = self.bow_tokenize(hypothesis)
        bow_inputs = np.concatenate([bow_premise, bow_hypothesis])

        return {
            "electra_input": electra_inputs,
            "bow_input": torch.tensor(bow_inputs, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }
        
def build_bow_vocab(dataset, vocab_size=10000):
    """
    Build a bag-of-words vocabulary based on word frequency in the dataset.
    """
    counter = Counter()
    for example in dataset:
        counter.update(example["premise"].lower().split())
        counter.update(example["hypothesis"].lower().split())
    most_common = counter.most_common(vocab_size)
    return {word: idx for idx, (word, _) in enumerate(most_common)}

def filter_invalid_labels(dataset):
    """
    Remove examples with invalid labels (label == -1) from the dataset.
    """
    return dataset.filter(lambda example: example["label"] != -1)

def prepare_snli_dataloaders(batch_size=32, max_seq_len=128, vocab_size=10000):
    # Load SNLI Dataset
    dataset = load_dataset("snli")

    # Filter out invalid labels
    dataset["train"] = filter_invalid_labels(dataset["train"])
    dataset["validation"] = filter_invalid_labels(dataset["validation"])
    dataset["test"] = filter_invalid_labels(dataset["test"])

    # Build BoW Vocabulary
    bow_vocab = build_bow_vocab(dataset["train"], vocab_size=vocab_size)

    # Initialize ELECTRA Tokenizer
    electra_tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    # Wrap Datasets
    train_dataset = SNLIDataset(dataset["train"], electra_tokenizer, bow_vocab, max_seq_len)
    val_dataset = SNLIDataset(dataset["validation"], electra_tokenizer, bow_vocab, max_seq_len)
    test_dataset = SNLIDataset(dataset["test"], electra_tokenizer, bow_vocab, max_seq_len)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for batch in train_loader:
        print("Electra Input:", batch["electra_input"])
        print("Electra Input Shapes:", {k: v.shape for k, v in batch["electra_input"].items()})
        print("BoW Input Shape:", batch["bow_input"].shape)
        print("Label Shape:", batch["label"].shape)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, bow_vocab