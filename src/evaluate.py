from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.train_mce import MixedCapacityEnsemble, BagOfWordsModel
from src.utils import compute_accuracy

def evaluate_all_models():
    # Load datasets
    dataset = load_dataset("snli")["test"]
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    # Load ELECTRA-small model
    electra_model = AutoModelForSequenceClassification.from_pretrained("./models/electra_small")
    print("Evaluating ELECTRA-small...")
    # Preprocessing and evaluation logic here...

    # Load MCE
    bow_vocab = {word: idx for idx, word in enumerate(["nobody", "outside", "cat", "dog", "favorite", "sad", "tall"])}
    low_capacity_model = BagOfWordsModel(len(bow_vocab), 300, 3)
    mce_model = MixedCapacityEnsemble(electra_model, low_capacity_model)
    print("Evaluating MCE...")
    # Preprocessing and evaluation logic here...

if __name__ == "__main__":
    evaluate_all_models()
