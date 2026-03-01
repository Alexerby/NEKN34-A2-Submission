import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from .._latex_tables import get_stars
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox


def get_dataset_metadata(series: pd.Series, id_label: str):
    """
    Returns physical characteristics of the dataset.
    """
    return {
        "ID": id_label,
        "Start Date": series.index[0].strftime("%Y-%m-%d"),  # pyright: ignore
        "End Date": series.index[-1].strftime("%Y-%m-%d"),  # pyright: ignore
        "Obs ($T$)": len(series),
    }


def get_descriptive_stats(data: np.ndarray):
    """
    Calculates key moments, AR(1) significance, and ARCH-LM.
    """
    mean = np.mean(data)
    std = np.std(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data, fisher=False)

    # Mean Model: AR(1) check
    # Fit an AR(1) to see if the first lag is significant
    ar_model = AutoReg(data, lags=1).fit()

    # index 1 is the first lag (index 0 is the constant)
    rho = ar_model.params[1] if len(ar_model.params) > 1 else 0
    rho_p = ar_model.pvalues[1] if len(ar_model.pvalues) > 1 else 1.0

    # Jarque-Bera Test
    jb_stat, jb_p = stats.jarque_bera(data)

    # ARCH-LM Test (Lag 1)
    arch_res = het_arch(data - mean)  # passing in residuals
    lm_stat, lm_p = arch_res[0], arch_res[1]

    return {
        "Mean": mean,
        "Std Dev": std,
        "Skew": skew,
        "Kurt": kurt,
        "AR(1)": f"{rho:.4f}{get_stars(rho_p)}",
        "JB-Stat": f"{jb_stat:.2f}{get_stars(jb_p)}",
        "ARCH-LM": f"{lm_stat:.2f}{get_stars(lm_p)}",
    }


def get_mean_model_diagnostics(data: np.ndarray, lags: int = 5):
    model = AutoReg(data, lags=1).fit()
    rho = model.params[1] if len(model.params) > 1 else 0
    p_val_rho = model.pvalues[1] if len(model.pvalues) > 1 else 1.0

    lb_df = acorr_ljungbox(data, lags=[lags], return_df=True)
    q_stat = lb_df["lb_stat"].iloc[0]
    q_p = lb_df["lb_pvalue"].iloc[0]

    return {
        "AR(1)": f"{rho:.4f}{get_stars(p_val_rho)}",
        f"Q({lags})": f"{q_stat:.2f}{get_stars(q_p)}",
    }
