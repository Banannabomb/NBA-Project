import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_margin_dotplot(input_csv="predictions.csv"):
    df = pd.read_csv(input_csv)

    if 'Margin' not in df.columns or 'team' not in df.columns:
        raise ValueError("CSV must contain 'Margin' and 'team' columns.")

    # Compute 75th percentile threshold
    threshold = df['Margin'].quantile(0.75)

    # Sort by margin
    df_sorted = df.sort_values(
        by="Margin", ascending=True).reset_index(drop=True)

    # Setup plot
    plt.figure(figsize=(14, 3))
    sns.scatterplot(data=df_sorted, x="Margin", y=[""]*len(df_sorted),
                    hue="Prediction", palette="coolwarm", s=100, legend=False)

    # Annotate teams
    for i, row in df_sorted.iterrows():
        plt.text(row['Margin'], 0, row['team'], fontsize=7,
                 ha='center', va='bottom', rotation=90)

    # Add threshold line
    plt.axvline(threshold, color='red', linestyle='--',
                label='75th Percentile Threshold')

    # Labels and layout
    plt.xlabel("SVM Margin")
    plt.title("Team Placement by Margin (Dot Plot)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_margin_dotplot("predictions/warriorsVbulls2010s.csv")
