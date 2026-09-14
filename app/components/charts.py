import pandas as pd
import streamlit as st


def strategy_comparison_chart(
    results_df: pd.DataFrame
):

    if results_df.empty:

        return

    chart_data = results_df.set_index(
        "strategy"
    )[
        [
            "profit",
            "revenue"
        ]
    ]

    st.bar_chart(
        chart_data
    )
