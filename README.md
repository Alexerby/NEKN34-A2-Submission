# NEKN34-A2-Submission

Replication code for the paper *The GARCH of the Rising Sun: Estimating Volatility in Yen-USD*.

## Setup

**Step 1:** Copy the sample config and set `project_root` to the root of your LaTeX project:
```bash
cp config.sample.json config.json
```

**Step 2:** Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Each script can be run independently. All outputs (LaTeX tables) are saved automatically to the configured LaTeX project root.

```bash
python scripts/data.py                    # Descriptive statistics
python scripts/models_dataset1.py         # Replication of Tse (1998)
python scripts/models_dataset2.py         # Replication of Tsui & Ho (2004)
python scripts/models_dataset_extended.py # Extended sample (2003-2023)
```
