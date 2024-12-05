import matplotlib.pyplot as plt
import string
from collections import Counter, defaultdict
from datasets import load_dataset
import pandas as pd
import numpy as np
from scipy.stats import norm

def add_annotations(ax, df, words, superscript, x_col, y_col):
    for word in words:
        point = df[df["word"] == word]
        if not point.empty:
            x = point[x_col].values[0]
            y = point[y_col].values[0]
            ax.annotate(
                f"{word}{superscript}", 
                (x, y), 
                fontsize=10, 
                xytext=(5, 5),
                textcoords="offset points", 
                ha="center"
            )
            
def conditional_colors(p_hat_values, counts, default_color, threshold_function):
    colors = []
    for p_hat, count in zip(p_hat_values, counts):
        threshold = threshold_function(count)
        if p_hat < threshold:
            colors.append("#D3D3D3")
        else:
            colors.append(default_color)
    return colors

if __name__ == "__main__":
    # Load and clean SNLI data (remove examples labeled -1)
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
    
    # Flatten tokens and count words
    all_tokens = [token for example in dataset['train']['combined'] for token in example]

    word_counts = Counter(all_tokens)
    word_counts_df = pd.DataFrame(word_counts.items(), columns=["word", "count"])
    word_counts_df = word_counts_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    print(word_counts_df.head())
    
    # Count by label
    label_word_counts = {
        "entailment": defaultdict(int),
        "contradiction": defaultdict(int),
        "neutral": defaultdict(int),
    }

    # Count word occurrences for each label
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

    # Ensure counts are integers
    word_counts_df[["entailment_count", "contradiction_count", "neutral_count"]] = word_counts_df[
        ["entailment_count", "contradiction_count", "neutral_count"]
    ].astype(int)
    
    # get p-hat and z-scores
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
    
    annotations = {
        "contradiction": ["sleeping", "cat", "nobody"],
        "neutral": ["tall", "sad", "favorite"],
        "entailment": ["outside", "people", "outdoors"]
    }

    rejection_threshold = lambda count: (z_star / (3 * np.sqrt(count))) + (1 / 3)
    n_values = np.logspace(np.log10(word_counts_df['count'].min()), np.log10(word_counts_df['count'].max()), 500)
    p_hat_threshold = (z_star / (3 * np.sqrt(n_values))) + (1 / 3)

    neutral_colors = conditional_colors(word_counts_df['p_hat_neutral'], word_counts_df['count'], "#F0E442", rejection_threshold)
    entailment_colors = conditional_colors(word_counts_df['p_hat_entailment'], word_counts_df['count'], "#56B4E9", rejection_threshold)
    contradiction_colors = conditional_colors(word_counts_df['p_hat_contradiction'], word_counts_df['count'], "#E69F00", rejection_threshold)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        word_counts_df['count'], 
        word_counts_df['p_hat_neutral'], 
        label='Neutral', 
        alpha=.8, 
        c=neutral_colors, 
        s=8
    )

    plt.scatter(
        word_counts_df['count'], 
        word_counts_df['p_hat_entailment'], 
        label='Entailment', 
        alpha=.8, 
        c=entailment_colors, 
        s=8
    )

    plt.scatter(
        word_counts_df['count'], 
        word_counts_df['p_hat_contradiction'], 
        label='Contradiction', 
        alpha=.8, 
        c=contradiction_colors, 
        s=8
    )

    plt.plot(
        n_values, 
        p_hat_threshold, 
        label=r"$\alpha = 0.01/23k$", 
        color="black", 
        linestyle="--", 
        linewidth=1.5
    )

    # Add annotations to the plot
    ax = plt.gca()  # Get the current axes
    add_annotations(ax, word_counts_df, annotations["contradiction"], r"$^c$", "count", "p_hat_contradiction")
    add_annotations(ax, word_counts_df, annotations["neutral"], r"$^n$", "count", "p_hat_neutral")
    add_annotations(ax, word_counts_df, annotations["entailment"], r"$^e$", "count", "p_hat_entailment")


    # Set log scale for x-axis
    plt.xscale("log")

    # Set y-axis limits to leave some space at the bottom
    plt.ylim(-0.02, 1.03)  # Extend slightly below 0
    plt.xlim(10, 10e5)  # Leave space on the left

    # Add labels, legend, and title
    plt.xlabel("n", fontsize=14)
    plt.ylabel(r"$\hat{p}(y | x_i)$", fontsize=14)
    plt.title("Artifact Statistics in SNLI", fontsize=16)
    plt.legend(title="", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Neutral', markersize=6, markerfacecolor="#F0E442"),
            plt.Line2D([0], [0], marker='o', color='w', label='Entailment', markersize=6, markerfacecolor="#56B4E9"),
            plt.Line2D([0], [0], marker='o', color='w', label='Contradiction', markersize=6, markerfacecolor="#E69F00"),
            plt.Line2D([0], [0], color="black", label=r"$\alpha = 0.01/23k$", linestyle="--", linewidth=1.5)
        ],
        fontsize=12,
        title="",
    )

    # Show the plot
    plt.savefig("artifact_statistics_snli.png", dpi=300, bbox_inches="tight")
    plt.show()

