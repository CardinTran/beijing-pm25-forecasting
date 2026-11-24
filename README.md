# Estimating Hourly PM2.5 in Beijing from Weather Data

Group 7, CSC 3730 Machine Learning and Data Analytics, Louisiana State University

Cardin Tran, Jimmy Rubio-Gonalez, Isaac Humphries, Vamsi Marrapu, Abdul Mutaal, Nhi Tran, Renee Vo, and David Zhong

We use weather and calendar data to estimate PM2.5 for the same hour. We compare seven models, from a mean baseline to tuned tree ensembles.

## Data

The original CSV is in `data/`. It has 43,824 hours. Of these, 2,067 are missing a PM2.5 reading and 41,757 can be used to score models. We build weather histories before dropping hours with missing readings. PM2.5 readings are used only as targets. [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) and [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

- Train on 2010 to 2012 using 24,418 hours.
- Choose model settings on 2013 using 8,678 hours. XGBoost also uses this year for early stopping.
- Refit on 2010 to 2013 using 33,096 hours.
- Score all models on the same 8,661 hours in 2014. Earlier versions of the project already reported some 2014 scores.

## Run the project

Use Python 3.12. On macOS, run `brew install libomp` for XGBoost. From the project folder, set up the environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3.12 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`, then run the same install command.

Run all 204 search candidates and the final experiments.

```sh
python main.py --mode full --jobs 4
```

Use the saved settings for a faster run. This still fits and scores the models. Running `python main.py` also uses saved settings.

```sh
python main.py --mode saved --jobs 4
```

Run the tests with `python -m unittest discover -s tests -v`.

## Results and report

The tables go in `results/` and the plots go in `report/figures/`. Add `--output-dir path/to/run` to save a separate run with its own tables and figures. Full mode writes the search tables. Saved mode leaves them in place.

- `model_comparison.csv` has MAE, RMSE, R² and sample counts.
- `best_hyperparameters.json` has the chosen settings. The grid search CSVs have every candidate and its 2013 RMSE.
- `predictions.csv` has timestamps, actual PM2.5 and model predictions.
- The other CSVs show how errors change with weather history, tuning, pollution level and season. They also include feature importance and uncertainty in the differences between models.

Read the [report](report/Group_7.pdf). To build it in Overleaf, upload the contents of `report/` and use `main.tex` with pdfLaTeX. If you have LaTeX installed locally, run `cd report && latexmk -pdf -jobname=Group_7 main.tex`.
