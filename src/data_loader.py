"""Load the UCI CSV."""

from pathlib import Path

import pandas as pd

# Expect the original UCI columns
REQUIRED_COLUMNS = [
    "No", "year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP",
    "PRES", "cbwd", "Iws", "Is", "Ir",
]


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Put the UCI CSV in data/."
        )
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {path}")
    try:
        # Read NA markers as missing values
        data = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError(f"Cannot read CSV: {path}") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("The dataset is empty.")
    return data