import os
from src.generate_artifact_statistics import generate_artifact_statistics
from src.train_models import train_all_models
from src.evaluate_models import evaluate_all_models
from datasets import load_dataset

def reproduce_results():
    print("Step 1: Generating artifact statistics...")
    dataset = load_dataset("snli")["train"]
    generate_artifact_statistics(dataset)

    print("Step 2: Training models...")
    train_all_models()

    print("Step 3: Evaluating models...")
    evaluate_all_models()

if __name__ == "__main__":
    reproduce_results()
