# Forecasting Hourly Air Pollution in Beijing Using Weather Data

We estimate hourly PM2.5 in Beijing using weather and time features. These are same-hour estimates rather than future-hour forecasts.

CSC 3730 — Machine Learning and Data Analytics

Louisiana State University — Group 7

## Members

- Jimmy Rubio-Gonalez
- Cardin Tran
- Isaac Humphries
- Vamsi Marrapu
- Abdul Mutaal
- Nhi Tran
- Renee Vo
- David Zhong

## Dataset and models

Use the [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) from 2010–2014. Place `PRSA_data_2010.1.1-2014.12.31.csv` in `data/`.

Models:

- Mean baseline
- Linear Regression
- Ridge Regression
- Random Forest Regression

Train on 2010–2012 and validate on 2013. Refit on 2010–2013 and test on 2014. Preprocessing learns from training data only. Rows without PM2.5 are dropped.

Source: Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C5JS49). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Setup

Install Python 3.11 and Git. Clone the repo:

```bash
git clone https://github.com/CardinTran/beijing-pm25-forecasting.git
cd beijing-pm25-forecasting
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation use `.venv\Scripts\activate.bat` in Command Prompt.

## Run

With `.venv` active run these commands from the repo folder:

```bash
python -m pip install -r requirements.txt
python main.py
```

To leave the environment:

```bash
deactivate
```

## Results

The program prints MAE and RMSE in µg/m³ plus R². It saves:

- `results/model_comparison.csv` — validation and test scores
- `results/figures/model_comparison.png` — test score comparison
- `results/figures/actual_vs_predicted.png` — predictions against actual 2014 values
- `results/figures/correlation_heatmap.png` — numeric correlations from 2010–2013

Each run replaces these outputs.