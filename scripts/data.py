import pandas as pd
from src.data_processor import get_dataset
from src.descriptives.plots import (
    plot_volatility_evidence,
    plot_distribution_comparison,
    plot_acf_pacf,
)
from src.descriptives.diagnostics import (
    get_descriptive_stats,
    get_dataset_metadata,
    get_mean_model_diagnostics,
)
from src.utils import save_output
from statsmodels.tsa.stattools import adfuller


def main():
    # =========================================================================
    # INITIALIZATION & DATA LOADING
    # =========================================================================
    dataset_ids = ["Dataset I", "Dataset II", "Extended"]
    metadata_list = []
    stats_list = []
    dataset_extended = None

    for ds_id in dataset_ids:
        series_raw = get_dataset(ds_id, transform="log")

        if isinstance(series_raw, pd.DataFrame):
            series = series_raw.iloc[:, 0]  # get first col
        else:
            series = series_raw

        if ds_id == "Extended":
            dataset_extended = series

        # =========================================================================
        # DIAGNOSTIC CALCULATIONS
        # =========================================================================
        metadata_list.append(get_dataset_metadata(series, ds_id))

        series_values = series.values
        stats_dict = get_descriptive_stats(series_values)  # pyright:ignore
        mean_diags = get_mean_model_diagnostics(series_values)  # pyright:ignore

        combined_stats = {**stats_dict, **mean_diags}
        combined_stats["ID"] = ds_id
        stats_list.append(combined_stats)

    # =========================================================================
    # VISUALIZATION PHASE
    # =========================================================================
    if dataset_extended is not None:
        plot_volatility_evidence(dataset_extended)
        save_output(None, "volatility_clustering", "figures", "descriptives")

        plot_distribution_comparison(
            dataset_extended, 8.494, title="Distributional Analysis"
        )
        save_output(None, "distribution_comparison", "figures", "descriptives")

        plot_acf_pacf(dataset_extended, nlags=20, squared=True)
        save_output(None, "acf_pacf", "figures", "descriptives")

        # =========================================================================
        # STATIONARITY TESTING
        # =========================================================================
        adf_stat, adf_pvalue, *_ = adfuller(dataset_extended, regression="c")
        print(f"ADF-stat: {adf_stat:.4f} with p-value {adf_pvalue:.4f}")

    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================
    df_metadata = pd.DataFrame(metadata_list).set_index("ID")
    df_stats = pd.DataFrame(stats_list).set_index("ID")

    # Utilizing the new caption and custom note features
    save_output(
        df_metadata,
        "dataset_metadata.tex",
        "tables",
        "diagnostics",
        caption="Summary of Dataset Metadata and Sample Periods",
        note="Start and end dates represent the available log-return series after synchronization.",
    )

    save_output(
        df_stats,
        "descriptives.tex",
        "tables",
        "diagnostics",
        caption="Descriptive Statistics and Mean Model Diagnostics",
        note=[
            r"Returns are defined as $r_t = (\ln S_t - \ln S_{t-1}) \times 100$.",
        ],
    )


if __name__ == "__main__":
    main()
