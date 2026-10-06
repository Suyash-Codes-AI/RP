# RP

A Python project for reproducing a steel-industry energy usage prediction workflow. The repository loads the dataset in `data/steel_industry.csv`, prepares time-based features, encodes categorical variables, applies feature-selection methods (Boruta and RFECV), and compares five regression models:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost
- LightGBM

The script saves model comparison metrics and selected feature outputs under the `results/` directory.

## Project goal

The objective is to reproduce and evaluate a predictive modeling pipeline for forecasting energy consumption (`Usage_kWh` / `Usage`) from operational and time-based variables in the steel industry dataset.

## Repository structure

```text
RP/
├── 01_reproduce_original_paper.py     # Main reproduction script
├── RP_PAPER.ipynb                     # Notebook version of the analysis
├── data/
│   └── steel_industry.csv            # Steel industry dataset
├── results/
│   ├── boruta_results.csv
│   ├── boruta_results_5models.csv
│   ├── encoded_features.csv
│   ├── final_selected_features_5models.csv
│   ├── five_model_comparison.csv
│   ├── rfe_results.csv
│   ├── rfe_results_5models.csv
│   └── replicate_steel_energy_paper.py
├── .gitignore.txt
└── README.md
```

## Data

The project uses a steel-industry operational dataset stored in:

- `data/steel_industry.csv`

The script validates time-related columns and prepares a cleaned dataset with fields including usage, reactive power, CO2 emissions, power factors, NSM, weekday/weekend status, and load type.

## Workflow

The main script (`01_reproduce_original_paper.py`) performs the following steps:

1. Reads and validates the dataset
2. Sorts records by timestamp and engineers derived time features
3. Encodes categorical variables using `OneHotEncoder`
4. Runs Boruta feature selection
5. Runs RFECV feature selection
6. Trains and evaluates five models
7. Saves metrics and selected feature lists to `results/`

## Getting started

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn boruta xgboost lightgbm jupyter
```

### Run the reproduction script

```bash
python 01_reproduce_original_paper.py
```

This will generate output files in the `results/` folder, including `five_model_comparison.csv`.

### Open the notebook

```bash
jupyter notebook RP_PAPER.ipynb
```

## Outputs

The script writes several CSV files into `results/`:

- `five_model_comparison.csv` — model comparison metrics such as RMSE, MAE, MAPE, and CV%
- `rfe_results_5models.csv` and `boruta_results_5models.csv` — feature selection rankings
- `final_selected_features_5models.csv` — final selected feature set

## Notes

- The repository appears to be a reproducibility-focused ML experiment rather than a packaged library.
- This project is implemented primarily in Python and is designed around Jupyter Notebook and scikit-learn-based modeling.

## License

No explicit license file is present in the repository at the moment.

## Author

Suyash-Codes-AI
