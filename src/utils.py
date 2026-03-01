from pathlib import Path
from typing import Any, Literal
import json
from ._latex_tables import export_to_latex
import pandas


def get_path(filename: str):
    return Path(__file__).parent.parent / "data" / filename


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config():
    root = get_project_root()
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("config.json not found in project root.")
    with open(config_path, "r") as f:
        return json.load(f)


def save_output(
    data: Any,
    filename: str,
    category: Literal["figures", "tables"],
    subfolder: str = "",
    root: Path | None = None,
    caption: str | None = None,
    note: str | None | list[str] = None,
):
    config = load_config()

    if root is None:
        config_root = config["paths"].get("project_root")
        root = Path(config_root) if config_root else get_project_root()

    base_rel_path = config["paths"].get(f"{category}_dir")
    if not base_rel_path:
        raise ValueError(f"Category '{category}' not found in config paths.")

    save_dir = root / base_rel_path / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / filename

    if category == "figures":
        import matplotlib.pyplot as plt

        fmt = config["settings"].get("plot_format", "png")
        dpi = config["settings"].get("dpi", 300)
        out_file = full_path.with_suffix(f".{fmt}")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {out_file}")

    elif category == "tables":
        if filename.endswith(".csv"):
            data.to_csv(full_path, index=True)
        elif filename.endswith(".tex"):
            # Now passing the optional caption and note to our improved exporter
            export_to_latex(data, full_path, caption=caption, note=note)
        print(f"Table saved to: {full_path}")


