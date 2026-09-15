from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_RESULTS = (
    ROOT /
    "data/experiments/"
    "model_comparison.csv"
)

CAMPAIGN_RESULTS = (
    ROOT /
    "data/experiments/"
    "campaign_strategy_results.csv"
)

OUTPUT_DIR = (
    ROOT /
    "docs/screenshots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def create_model_comparison():

    if not MODEL_RESULTS.exists():

        print(
            "Model comparison file not found:"
        )

        print(MODEL_RESULTS)

        return

    df = pd.read_csv(
        MODEL_RESULTS
    )

    required = [
        "model",
        "roc_auc",
        "pr_auc"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            f"Missing columns: {missing}"
        )

        return

    plot_data = df.set_index(
        "model"
    )[[
        "roc_auc",
        "pr_auc"
    ]]

    ax = plot_data.plot(
        kind="bar",
        figsize=(9, 6)
    )

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "model_performance.png",
        dpi=200
    )

    plt.close()


def create_campaign_comparison():

    if not CAMPAIGN_RESULTS.exists():

        print(
            "Campaign results file not found:"
        )

        print(CAMPAIGN_RESULTS)

        return

    df = pd.read_csv(
        CAMPAIGN_RESULTS
    )

    if "strategy" not in df.columns:

        print(
            "strategy column missing."
        )

        return

    if "profit" in df.columns:

        profit = (
            df.set_index("strategy")
            ["profit"]
        )

        ax = profit.plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_title(
            "Campaign Strategy — Profit"
        )

        ax.set_ylabel(
            "Profit"
        )

        ax.set_xlabel(
            "Strategy"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            "campaign_profit.png",
            dpi=200
        )

        plt.close()

    if "roi" in df.columns:

        roi = (
            df.set_index("strategy")
            ["roi"]
        )

        ax = roi.plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_title(
            "Campaign Strategy — ROI"
        )

        ax.set_ylabel(
            "ROI"
        )

        ax.set_xlabel(
            "Strategy"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            "campaign_roi.png",
            dpi=200
        )

        plt.close()


def main():

    print(
        "Generating portfolio visuals..."
    )

    create_model_comparison()

    create_campaign_comparison()

    print(
        "Visual generation complete."
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
