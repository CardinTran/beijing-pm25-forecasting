# Forecasting Hourly Air Pollution in Beijing Using Weather Data

We're exploring how well weather and time-related data can estimate hourly PM2.5 levels in Beijing. We plan to compare a simple baseline with Linear Regression, Ridge Regression, and Random Forest Regression.

The project title says forecasting, but using weather from the same hour estimates pollution for that hour. We'll need to define a prediction horizon and available inputs before making claims about future-hour predictions. For now, we're just setting up the Python environment.

**Course:** CSC 3730 — Machine Learning and Data Analytics, Louisiana State University

**Group:** Group 7

## Members

- Jimmy Rubio-Gonalez
- Cardin Tran
- Isaac Humphries
- Vamsi Marrapu
- Abdul Mutaal
- Nhi Tran
- Renee Vo
- David Zhong

## Dataset

We'll use the [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data), which covers hourly pollution and weather observations from 2010 through 2014. The dataset hasn't been downloaded as part of this setup.

Source: Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C5JS49). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Setup

Install Python 3.11 and Git, then clone the repo:

```bash
git clone https://github.com/CardinTran/beijing-pm25-forecasting.git
cd beijing-pm25-forecasting
```

On macOS or Linux, create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

With the environment active, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

We're starting with NumPy, pandas, scikit-learn, Matplotlib, and seaborn. The versions in `requirements.txt` keep our starting environments consistent.

To leave the virtual environment:

```bash
deactivate
```

Activate it again whenever you return to the project. The `.venv` folder stays local and isn't committed to Git.
