import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_accuracy(predictions, labels):
    """
    Compute accuracy given predictions and labels.
    """
    return accuracy_score(labels, predictions)

def compute_precision_recall_f1(predictions, labels, average="weighted"):
    """
    Compute precision, recall, and F1 score.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average)
    return {"precision": precision, "recall": recall, "f1": f1}

def tokenize_texts(tokenizer, premises, hypotheses, max_seq_len=128):
    """
    Tokenize premise-hypothesis pairs using a tokenizer.
    """
    return tokenizer(
        premises,
        hypotheses,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt"
    )

def load_model(model_class, model_path, config=None):
    """
    Load a model from a given path.
    """
    if config:
        return model_class.from_pretrained(model_path, config=config)
    return model_class.from_pretrained(model_path)

def split_dataset(dataset, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split a dataset into training, validation, and test sets.
    """
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_end = int(split_ratios[0] * len(dataset))
    val_end = train_end + int(split_ratios[1] * len(dataset))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_set = dataset.select(train_indices)
    val_set = dataset.select(val_indices)
    test_set = dataset.select(test_indices)

    return train_set, val_set, test_set

def create_dataloader(dataset, batch_size=16, shuffle=False):
    """
    Create a PyTorch DataLoader from a dataset.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_model_checkpoint(model, tokenizer, output_dir):
    """
    Save model and tokenizer to a specified directory.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_model_checkpoint(model_class, tokenizer_class, checkpoint_dir):
    """
    Load model and tokenizer from a specified checkpoint directory.
    """
    model = model_class.from_pretrained(checkpoint_dir)
    tokenizer = tokenizer_class.from_pretrained(checkpoint_dir)
    print(f"Model loaded from {checkpoint_dir}")
    return model, tokenizer

def progress_bar(iterable, total=None):
    """
    Simple progress bar utility for loops.
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total)
    except ImportError:
        return iterable  # Fallback without tqdm

def set_seed(seed=1998):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

