import pandas as pd
from pathlib import Path
import re

PARAM_MAP = {
    "Const": r"$\mu$",
    "Const_std": "",
    "y[1]": r"$\rho$",
    "y[1]_std": "",
    "omega": r"$\eta$",
    "omega_std": "",
    "alpha[1]": r"$\alpha$",
    "alpha[1]_std": "",
    "beta[1]": r"$\beta_G$",
    "beta[1]_std": "",
    "beta": r"$\beta_F$",
    "beta_std": "",
    "d": r"$d$",
    "d_std": "",
    "phi": r"$\phi$",
    "phi_std": "",
    "gamma[1]": r"$\gamma$",
    "gamma[1]_std": "",
    "delta": r"$\delta$",
    "delta_std": "",
}

DESIRED_ORDER = [
    "Const",
    "Const_std",
    "y[1]",
    "y[1]_std",
    "omega",
    "omega_std",
    "alpha[1]",
    "alpha[1]_std",
    "beta[1]",
    "beta[1]_std",
    "gamma[1]",
    "gamma[1]_std",
    "delta",
    "delta_std",
    "beta",
    "beta_std",
    "d",
    "d_std",
    "phi",
    "phi_std",
]


def get_stars(p_value):
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    return ""


def _get_deterministic_label(filename_stem: str) -> str:
    label = filename_stem.lower().replace("_", "-").replace(" ", "-")
    label = re.sub(r"[^a-z0-9\-]", "", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return f"tab:{label}"


def _get_significance_note(
    df: pd.DataFrame, custom_note: str | list[str] | None = None
) -> str:
    has_stars = df.map(lambda x: "*" in str(x)).any().any()

    if not has_stars and not custom_note:
        return ""

    # Collect all notes into a single list to ensure consecutive numbering
    raw_notes = []

    if has_stars:
        raw_notes.append(
            r"\textit{Note:} ***, **, and * denote significance at the 1\%, 5\%, and 10\% levels, respectively."
        )

    if custom_note:
        if isinstance(custom_note, str):
            raw_notes.append(custom_note)
        else:
            raw_notes.extend(custom_note)

    # Generate the LaTeX items with iteration numbers
    note_items = []
    for i, n in enumerate(raw_notes, 1):
        clean_n = n.replace("%", r"%")
        note_items.append(f"\\item Note {i}: {clean_n}")

    return (
        "\n"
        + r"\vspace{0.1cm}"
        + "\n"
        + r"\begin{tablenotes}"
        + "\n"
        + r"\small"
        + "\n"
        + "\n".join(note_items)
        + "\n"
        + r"\end{tablenotes}"
    )


def _apply_parameter_mapping(df: pd.DataFrame) -> pd.DataFrame:
    # Explicitly map every key produced by the ARCH package
    param_map = {
        "Const": r"$\mu$",
        "Const_std": "",
        "y[1]": r"$\rho$",
        "y[1]_std": "",
        "omega": r"$\eta$",
        "omega_std": "",
        "alpha[1]": r"$\alpha$",
        "alpha[1]_std": "",
        "gamma[1]": r"$\gamma$",
        "gamma[1]_std": "",
        "beta[1]": r"$\beta$",
        "beta[1]_std": "",
        # Catch for models where arch uses 'beta' instead of 'beta[1]'
        "beta": r"$\beta$",
        "beta_std": "",
        "d": r"$d$",
        "d_std": "",
        "phi[1]": r"$\phi$",
        "phi[1]_std": "",
        "phi": r"$\phi$",
        "phi_std": "",
        "delta": r"$\delta$",
        "delta_std": "",
    }

    df.index = df.index.map(lambda x: param_map.get(x, x))
    return df


def export_to_latex(
    df: pd.DataFrame,
    full_path: Path,
    caption: str | None = None,
    note: str | list[str] | None = None,
):
    # 1. Apply the parameter cleaning first
    df = _apply_parameter_mapping(df)

    table_label = _get_deterministic_label(full_path.stem)

    # 2. Reset index but rename the 'index' column to empty string
    df_to_export = df.reset_index()
    df_to_export.rename(columns={"index": ""}, inplace=True)

    styler = df_to_export.style
    styler.hide(axis="index")
    styler.format(precision=4, na_rep="---")

    # Column alignment: left for the param names, center for the results
    col_fmt = "l" + "c" * (len(df_to_export.columns) - 1)

    # Generate the string
    latex_string = styler.to_latex(
        caption=caption or full_path.stem.replace("_", " ").title(),
        label=table_label,
        position="htbp",
        column_format=col_fmt,
        hrules=True,
    )

    # 3. Post-processing for Threeparttable & Tabularx
    latex_string = latex_string.replace(
        r"\begin{table}[htbp]",
        f"\\begin{{table}}[htbp]\n\\centering\n\\begin{{threeparttable}}",
    )
    latex_string = latex_string.replace(
        r"\end{table}", f"\\end{{threeparttable}}\n\\end{{table}}"
    )

    target_search = f"\\begin{{tabular}}{{{col_fmt}}}"
    tabularx_header = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_fmt}}}"
    )
    latex_string = latex_string.replace(target_search, tabularx_header)
    latex_string = latex_string.replace(r"\end{tabular}", r"\end{tabularx}")

    # 4. Inject Notes block
    note_block = _get_significance_note(df, custom_note=note)
    if note_block:
        latex_string = latex_string.replace(
            r"\end{tabularx}", f"\\end{{tabularx}}\n{note_block}"
        )

    with open(full_path, "w") as f:
        f.write(latex_string)


def format_coef_std(fit_result):
    """
    Extracts coefficients and robust standard errors,
    returning a series with standard errors in parentheses below coefficients.
    """
    params = fit_result.params
    std_errs = fit_result.std_err

    formatted = {}
    for name in params.index:
        formatted[name] = f"{params[name]:.4f}"
        formatted[f"{name}_std"] = f"({std_errs[name]:.4f})"

    return pd.Series(formatted)
