"""
visualize.py
Generates plots for BMI-PCOS correlation and model comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_bmi_pcos_correlation(df, target_col='PCOS (Y/N)'):
    """Bar chart comparing average BMI across PCOS diagnosis groups."""
    plt.figure(figsize=(6, 4))
    df.groupby(target_col)['BMI'].mean().plot(kind='bar', color=['#4CAF50', '#F44336'])
    plt.title("Average BMI by PCOS Diagnosis")
    plt.xlabel("PCOS (0 = No, 1 = Yes)")
    plt.ylabel("Average BMI")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("bmi_pcos_correlation.png")
    plt.show()


def plot_model_accuracies(results: dict):
    """Bar chart comparing accuracy of all trained models."""
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=list(results.keys()),
        y=list(results.values()),
        hue=list(results.keys()),
        palette="viridis",
        legend=False
    )
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.show()