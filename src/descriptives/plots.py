import matplotlib.pyplot as plt
from .._apa_style import apply_apa_style, cleanup_axis
from scipy.stats import norm, t
import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf


def plot_overview(series, title):
    apply_apa_style()
    plt.figure(figsize=(8, 4))
    series.plot()
    cleanup_axis("Date", "Exchange Rate (JPY/USD)", title)


def plot_volatility_evidence(returns):
    apply_apa_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    returns.plot(ax=ax1)
    ax1.set_title("Log Returns", loc="left", fontsize=10)
    ax1.set_ylabel("$r_t$")

    (returns**2).plot(ax=ax2)
    ax2.set_title("Squared Returns", loc="left", fontsize=10)
    ax2.set_ylabel("$\epsilon_t^2$")

    plt.xlabel("Date")
    plt.tight_layout()


def plot_distribution_comparison(series, kurtosis, title="Distributional Comparison"):
    apply_apa_style()

    mean = series.mean()
    std = series.std()

    excess_k = kurtosis - 3
    nu = 4 + (6 / excess_k)
    s = std * np.sqrt((nu - 2) / nu)

    # These define your "viewing window"
    x_min = mean - 4 * std
    x_max = mean + 4 * std
    x = np.linspace(x_min, x_max, 1000)

    y_norm = norm.pdf(x, mean, std)
    y_t = t.pdf(x, nu, mean, s)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    plt.hist(
        series,
        bins=100,  # Increased bins for better resolution with outliers
        range=(x_min, x_max),  # Clip the histogram calculation range
        density=True,
        alpha=0.2,
        color="gray",
        label="Empirical Histogram",
    )

    series.plot.kde(color="black", linewidth=1, label="Empirical KDE", alpha=0.8)

    plt.plot(
        x,
        y_norm,
        label=f"Normal ($\mu$={mean:.3f}, $\sigma$={std:.3f})",
        linestyle="--",
        color="tab:red",
        alpha=0.8,
    )

    plt.plot(
        x,
        y_t,
        label=f"Student's t ($\\nu$={nu:.2f}, $s$={s:.3f})",
        color="tab:blue",
        linewidth=1.5,
    )

    # Clip the actual axes to ignore extreme outliers
    ax.set_xlim(x_min, x_max)

    cleanup_axis(ax, title, "Density")

    plt.xlabel("Returns")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()


def plot_acf_pacf(data, nlags=40, squared=False, title=None):
    apply_apa_style()
    
    label = "Squared Returns" if squared else "Returns"
    if title is None:
        title = f"Autocorrelation Analysis: {label}"
        
    analysis_data = data**2 if squared else data
    
    acf_vals = sm_acf(analysis_data, nlags=nlags)
    pacf_vals = sm_pacf(analysis_data, nlags=nlags, method="ywm")
    lags = np.arange(nlags + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    conf = 1.96 / np.sqrt(len(data))

    # We keep a small negative floor to show the lower confidence interval
    # while capping the top at 0.2 to zoom in.
    y_min = -0.05 if squared else -0.15
    y_max = 0.2

    # --- ACF Plot ---
    ax1.bar(lags, acf_vals, width=0.6, color="tab:blue", alpha=0.8)
    ax1.axhline(conf, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax1.axhline(-conf, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylim(y_min, y_max) 
    cleanup_axis(ax1, title, "ACF")

    # --- PACF Plot ---
    ax2.bar(lags, pacf_vals, width=0.6, color="tab:red", alpha=0.8)
    ax2.axhline(conf, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax2.axhline(-conf, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(y_min, y_max)
    cleanup_axis(ax2, "", "PACF")
    
    plt.xlabel("Lags")
    plt.tight_layout()
