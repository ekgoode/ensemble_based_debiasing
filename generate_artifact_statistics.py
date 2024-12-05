import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset

def generate_artifact_statistics(dataset, output_path="artifact_statistics_snli.png"):
    """
    Generate and save artifact statistics for the dataset.
    """
    counter = Counter()
    for example in dataset:
        counter.update(example["premise"].split())
        counter.update(example["hypothesis"].split())
    
    # Example artifact statistics (frequency plot of top words)
    most_common = counter.most_common(20)
    words, counts = zip(*most_common)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title("Dataset Artifacts in SNLI")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    dataset = load_dataset("snli")["train"]
    generate_artifact_statistics(dataset)
