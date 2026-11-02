# Evidently AI Labs: Data and Model Monitoring

Two notebooks that use [Evidently](https://www.evidentlyai.com/) (`evidently==0.7.0`) to detect data drift and track model quality.

| Notebook | What it covers |
|---|---|
| `Lab1.ipynb` | Baseline. Drift report on the Adult census dataset, split on `education` so drift is guaranteed. |
| `Lab2.ipynb` | Monitoring a deployed regression model on the Bike Sharing Demand dataset: monthly batches, drift and performance alerts, test suites, fault injection, retraining. |

## Requirements

* **Python 3.11 or 3.12.** Tested on Python 3.12 (Windows 11). Python 3.11 should work but has not been tested.
  * Python 3.13 does not work (confirmed): `evidently 0.7.0` needs `numpy<2.1`, which has no wheels there, so pip tries to compile it and fails. Newer versions are expected to fail the same way but were not tested.
* **pandas below 3.** `requirements.txt` pins `pandas==2.3.3`. With pandas 3.x, `import evidently` fails.
* **Internet access on the first run.** The datasets are downloaded from OpenML (about 20 seconds) and cached in `~/scikit_learn_data`. Later runs work offline.
* The install pulls in JupyterLab and many dependencies, so allow several minutes and a few hundred MB of disk.

## Setup

Run these from this folder (`Labs/Data_Labs/Data_Monitoring/Evidently_AI`).

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open `Lab2.ipynb` and choose **Run > Run All Cells**. The first code cell checks your Python and pandas versions and stops with a clear message if they are unsupported.

To run without opening Jupyter:
```bash
jupyter nbconvert --to notebook --execute Lab2.ipynb --output Lab2_executed.ipynb
```

## What you get

Running `Lab2.ipynb` creates a `reports/` folder (ignored by git) containing:

* `batch_2011-07.html`: the interactive Evidently drift and regression report for one batch (about 4 MB).
* `monitoring_timeline.png`: drift share, MAE and actual-vs-predicted demand over 18 months.

Expected results, which the notebook reproduces because the model and the data split are seeded:

* Reference MAE is about 45.8 bikes per hour.
* The data alert fires in 17 of 18 months, and the performance alert first fires in Feb 2012.
* In the fault-injection table the control batch has no drifted columns, and each injected fault flags its own column.

The numbers should match to the printed precision on the same package versions. Other versions of scikit-learn or Evidently may shift them slightly.

## Verification

`Lab1.ipynb` and `Lab2.ipynb` were run end to end on Windows 11 with Python 3.12, in a brand-new virtual environment created only from `requirements.txt`, with an empty dataset cache. Both finished with no errors and `Lab2.ipynb` reproduced the numbers above.

Not tested: Python 3.11, macOS and Linux (the setup commands for those are the standard equivalents), and opening the notebooks in JupyterLab (the runs used `nbconvert`).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Unable to detect GNU compiler type` or a numpy build error during `pip install` | You are on Python 3.13 or newer. Use 3.11 or 3.12. |
| `ImportError: cannot import name 'infer_dtype_from_object'` | pandas 3.x is installed. Run `pip install -r requirements.txt` in a clean environment. |
| `RuntimeError: Python ... is not supported` from the first cell | Same as above: use Python 3.11 or 3.12. |
| The data-loading cell hangs or fails | OpenML is unreachable. Check your connection and rerun the cell. Once cached, it works offline. |
| The report cell shows only `<evidently.core.report.Snapshot ...>` | Your front end is not rendering HTML. Use JupyterLab or VS Code, or open the saved file in `reports/`. |

## Notes

* `Lab1.ipynb` no longer uses Evidently Cloud, so no account or API token is needed.
* In `Lab2.ipynb`, compare per-column drift flags and not raw drift scores. Evidently uses p-values on small batches and distances on large ones, so the raw scores are not comparable.
