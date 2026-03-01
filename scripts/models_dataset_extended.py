import pandas as pd
from src._latex_tables import DESIRED_ORDER, PARAM_MAP, format_coef_std
from src.data_processor import get_dataset
from arch import arch_model
from src.utils import save_output


def estimate_models(data):
    results = {}
    # GARCH(1,1) - three distributions
    results["GARCH-N"] = arch_model(
        data, mean="AR", lags=1, vol="GARCH", p=1, o=0, q=1, dist="normal"
    )
    results["GARCH-t"] = arch_model(
        data, mean="AR", lags=1, vol="GARCH", p=1, o=0, q=1, dist="t"
    )
    results["GARCH-G"] = arch_model(
        data, mean="AR", lags=1, vol="GARCH", p=1, o=0, q=1, dist="ged"
    )
    # FIGARCH(1,d,1) - three distributions
    results["FIGARCH-N"] = arch_model(
        data, mean="AR", lags=1, vol="FIGARCH", p=1, q=1, dist="normal"
    )
    results["FIGARCH-t"] = arch_model(
        data, mean="AR", lags=1, vol="FIGARCH", p=1, q=1, dist="t"
    )
    results["FIGARCH-G"] = arch_model(
        data, mean="AR", lags=1, vol="FIGARCH", p=1, q=1, dist="ged"
    )
    return results


def main():
    series = get_dataset("Extended", transform="log")
    data = series.to_numpy()
    print("Estimating extended models...")
    model_fits = {
        name: model.fit(disp="off") for name, model in estimate_models(data).items()
    }
    table_data = {
        model_name: format_coef_std(fit) for model_name, fit in model_fits.items()
    }
    df_results = pd.DataFrame(table_data)
    existing_order = [k for k in DESIRED_ORDER if k in df_results.index]
    df_results = df_results.loc[existing_order]
    df_results.index = [PARAM_MAP.get(idx, idx) for idx in df_results.index]

    # Append AIC and BIC rows
    df_results.loc["AIC"] = {name: f"{fit.aic:.2f}" for name, fit in model_fits.items()}
    df_results.loc["BIC"] = {name: f"{fit.bic:.2f}" for name, fit in model_fits.items()}

    save_output(
        df_results,
        "estimation_results_extended.tex",
        "tables",
        "models",
        caption="Estimation Results for Extended Dataset (2003--2023)",
        note=[
            "Standard errors in parentheses are \\textcite{bollerslev_woolridge1996} robust standard errors.",
            "All models estimated via QMLE alongside an AR(1) mean equation.",
            "Column suffixes denote the error distribution: -N (Gaussian), -t (Student's $t$), -G (GED).",
        ],
    )


if __name__ == "__main__":
    main()
