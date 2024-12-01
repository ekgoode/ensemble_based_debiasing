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

class BoWModel(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_dim=128):
        super(BoWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        bow_rep = embedded.mean(dim=1)
        logits = self.fc(bow_rep)
        return logits


class HighCapacityModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(HighCapacityModel, self).__init__()
        self.electra = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

class MixedCapacityEnsemble(nn.Module):
    def __init__(self, high_capacity_model, low_capacity_model, num_classes, class_prior=None, loss_weight=1.0):
        super(MixedCapacityEnsemble, self).__init__()
        self.high_capacity_model = high_capacity_model
        self.low_capacity_model = low_capacity_model
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        
        if class_prior is None:
            self.class_prior = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        else:
            self.class_prior = nn.Parameter(torch.log(torch.tensor(class_prior)), requires_grad=False)

    def forward(self, high_input, low_input, labels=None):
        # Forward pass through high-capacity model
        high_output = self.high_capacity_model(**high_input)
        high_logits = high_output.logits  # Access logits
        
        # Forward pass through low-capacity model
        low_logits = self.low_capacity_model(low_input)
        
        # Compute ensemble logits
        ensemble_logits = high_logits + low_logits + self.class_prior

        if labels is not None:
            # Compute losses
            ensemble_loss = F.cross_entropy(ensemble_logits, labels)
            high_loss = F.cross_entropy(high_logits, labels)
            low_loss = F.cross_entropy(low_logits, labels)
            
            # Weighted loss
            total_loss = ensemble_loss + self.loss_weight * low_loss
            return total_loss, ensemble_logits, high_logits, low_logits
        else:
            return ensemble_logits, high_logits, low_logits

        
# Step 1: Define the SNLI Dataset Wrapper
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

# Step 2: Build the BoW Vocabulary
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

# Step 3: Prepare Data Loaders
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

def train_mixed_capacity_ensemble(model, dataloader, optimizer, criterion, w=1.0, device="cuda"):
    """
    Args:
        model: MixedCapacityEnsemble model.
        dataloader: PyTorch DataLoader for SNLI data.
        optimizer: Optimizer for model training.
        criterion: Loss function.
        w: Weight for low-capacity loss.
        device: "cuda" or "cpu".
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        electra_input = {k: v.to(device) for k, v in batch["electra_input"].items()}
        bow_input = batch["bow_input"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        ensemble_probs, low_probs, high_probs = model(electra_input, bow_input)

        # Compute losses
        ensemble_loss = criterion(ensemble_probs, labels)
        low_loss = criterion(low_probs, labels)
        high_loss = criterion(high_probs, labels)

        # Total loss
        loss = ensemble_loss + w * low_loss + high_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Step 3: Evaluation Loop
def evaluate_mixed_capacity_ensemble(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            electra_input = {k: v.to(device) for k, v in batch["electra_input"].items()}
            bow_input = batch["bow_input"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            ensemble_probs, _, _ = model(electra_input, bow_input)

            # Compute loss
            loss = criterion(ensemble_probs, labels)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(ensemble_probs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total

# Step 4: Main Training Script
def train():
    # Hyperparameters
    batch_size = 32
    max_seq_len = 128
    vocab_size = 10000
    num_classes = 3
    learning_rate = 3e-5
    num_epochs = 3
    weight_low_loss = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data loaders
    train_loader, val_loader, test_loader, bow_vocab = prepare_snli_dataloaders(
        batch_size=batch_size, max_seq_len=max_seq_len, vocab_size=vocab_size
    )

    # Define models
    electra_model = AutoModelForSequenceClassification.from_pretrained(
        "google/electra-small-discriminator", num_labels=num_classes
    ).to(device)

    bow_model = nn.Sequential(
        nn.Linear(vocab_size * 2, 300),  # BoW input size is vocab_size * 2 (premise + hypothesis)
        nn.ReLU(),
        nn.Linear(300, num_classes)
    ).to(device)

    # Mixed Capacity Ensemble
    model = MixedCapacityEnsemble(electra_model, bow_model, num_classes).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_mixed_capacity_ensemble(model, train_loader, optimizer, criterion, w=weight_low_loss, device=device)
        val_loss, val_acc = evaluate_mixed_capacity_ensemble(model, val_loader, criterion, device=device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"mixed_capacity_ensemble_epoch_{epoch + 1}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_epoch_{epoch + 1}.pt"))
    # Test evaluation
    test_loss, test_acc = evaluate_mixed_capacity_ensemble(model, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Run the training
train()
