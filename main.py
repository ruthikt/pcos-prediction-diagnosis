"""
main.py
Entry point: runs the full PCOS prediction pipeline.
"""

from data_preprocessing import prepare_dataset
from train_models import split_data, train_all_models
from visualize import plot_bmi_pcos_correlation, plot_model_accuracies


def main():
    filepath = "data/PCOS_data.csv"

    print("Loading and preprocessing data...")
    X, y, df = prepare_dataset(filepath)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training models...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\nGenerating visualizations...")
    plot_bmi_pcos_correlation(df)
    plot_model_accuracies(results)

    print("\nPipeline complete. Results summary:")
    for name, acc in results.items():
        print(f"  {name}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()