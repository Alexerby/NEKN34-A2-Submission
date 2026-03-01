import matplotlib.pyplot as plt


def apply_apa_style():
    """Clean, high-contrast academic style without layout bloat."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "DejaVu Serif",
                "serif",
            ],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def cleanup_axis(ax, title, ylabel):
    """Utility to set titles and labels without overlapping."""
    ax.set_title(title, loc="left", pad=15)
    ax.set_ylabel(ylabel)
    ax.grid(False)
