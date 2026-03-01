# models_dataset2.py

import pandas as pd
from arch import arch_model
from src.data_processor import get_dataset
from src.utils import save_output
from src._latex_tables import PARAM_MAP, DESIRED_ORDER, format_coef_std




def estimate_replication_models(data):
    results = {}

    # APARCH(1,1): Set o=1 to estimate gamma
    results["APARCH"] = arch_model(
        data, mean="AR", lags=1, vol="APARCH", p=1, o=1, q=1
    ).fit(disp="off", cov_type="robust")

    # AGARCH(1,1): Asymmetric, implemented via GJR-GARCH
    results["AGARCH"] = arch_model(
        data, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, power=2.0
    ).fit(disp="off", cov_type="robust")

    return results


def main():
    series = get_dataset("Dataset II", transform="log")
    data = series.to_numpy()

    print(f"Estimating replication models for Dataset II...")
    model_fits = estimate_replication_models(data)

    table_data = {
        model_name: format_coef_std(fit) for model_name, fit in model_fits.items()
    }
    df_results = pd.DataFrame(table_data)


    existing_order = [k for k in DESIRED_ORDER if k in df_results.index]
    df_results = df_results.loc[existing_order]


    df_results.index = [PARAM_MAP.get(idx, idx) for idx in df_results.index]

    save_output(
        df_results,
        "replication_results_d2.tex",
        "tables",
        "models",
        caption="Replication Results for Dataset II",
        note=[
            r"Standard errors in parentheses are \textcite{bollerslev_woolridge1996} robust standard errors.",
            "All models estimated via QMLE assuming a normal likelihood.",
        ],
    )
    print("Results exported to tables/models/replication_results_d2.tex")


if __name__ == "__main__":
    main()
