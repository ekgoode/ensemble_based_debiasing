from src.train_electra import train_electra
from src.train_mce import train_mce
from datasets import load_dataset
from transformers import AutoTokenizer
from src.utils import save_model_checkpoint

def train_all_models():
    # Load dataset
    dataset = load_dataset("snli")
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    # Train ELECTRA-small
    print("Training ELECTRA-small model...")
    electra_trainer = train_electra(
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir="./models/electra_small"
    )
    save_model_checkpoint(electra_trainer.model, tokenizer, "./models/electra_small")

    # Train MCE
    print("Training Mixed Capacity Ensemble (MCE)...")
    bow_vocab = {word: idx for idx, word in enumerate(["nobody", "outside", "cat", "dog", "favorite", "sad", "tall"])}
    train_mce(
        dataset=dataset,
        tokenizer=tokenizer,
        bow_vocab=bow_vocab,
        output_dir="./models/mce"
    )

if __name__ == "__main__":
    train_all_models()
