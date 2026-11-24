import contextlib
import io
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import (
    EXPANDED_FEATURES, FEATURES, HISTORY_COLUMNS, HISTORY_WINDOWS, TARGET,
    make_preprocessor, prepare_data,
)

DATASET = Path(__file__).resolve().parents[1] / "data/PRSA_data_2010.1.1-2014.12.31.csv"


class ChronologicalPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_dataset(DATASET)
        cls.weather = cls.raw.set_index(pd.to_datetime(cls.raw[["year", "month", "day", "hour"]]))
        cls.training, cls.validation, cls.testing = cls.prepare(cls.raw)

    @staticmethod
    def prepare(raw):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_data(raw)

    def test_counts_and_input_allowlist(self):
        self.assertEqual(len(self.raw), 43824)
        self.assertEqual(self.raw[TARGET].isna().sum(), 2067)
        splits = [self.training, self.validation, self.testing]
        self.assertEqual([len(split) for split in splits], [24418, 8678, 8661])
        self.assertEqual([sorted(set(split.index.year)) for split in splits],
                         [[2010, 2011, 2012], [2013], [2014]])
        self.assertEqual(len(set(FEATURES)), 10)
        self.assertEqual(len(set(EXPANDED_FEATURES)), 23)
        self.assertTrue(set(FEATURES).issubset(EXPANDED_FEATURES))
        self.assertTrue({"No", "year", "day", TARGET}.isdisjoint(EXPANDED_FEATURES))

    def test_windows_include_unlabeled_hours_and_cross_year_boundaries(self):
        cases = [self.training.index[0], self.validation.index[0], self.testing.index[0]]
        labeled = pd.concat([self.training, self.validation, self.testing])
        first_window = self.weather.loc[:cases[0]]
        self.assertTrue(first_window[TARGET].isna().any())
        for timestamp in cases:
            row = labeled.loc[timestamp]
            past = self.weather.loc[:timestamp]
            for column in HISTORY_COLUMNS:
                self.assertAlmostEqual(row[f"{column}_change_3h"], past[column].iloc[-1] - past[column].iloc[-4])
                for hours in HISTORY_WINDOWS:
                    self.assertAlmostEqual(row[f"{column}_mean_{hours}h"], past[column].tail(hours).mean())
            self.assertAlmostEqual(row["Ir_total_24h"], past["Ir"].tail(24).sum())

    def test_no_future_weather_or_target_enters_features(self):
        changed = self.raw.sample(frac=1, random_state=42).copy()
        cutoff = pd.Timestamp("2013-01-01 12:00")
        timestamps = pd.to_datetime(changed[["year", "month", "day", "hour"]])
        changed.loc[timestamps > cutoff, HISTORY_COLUMNS + ["Ir"]] += 1000
        changed[TARGET] *= 2
        training, validation, _ = self.prepare(changed)
        pd.testing.assert_frame_equal(training[EXPANDED_FEATURES], self.training[EXPANDED_FEATURES])
        pd.testing.assert_frame_equal(validation.loc[:cutoff, EXPANDED_FEATURES],
                                      self.validation.loc[:cutoff, EXPANDED_FEATURES])

    def test_encoder_handles_categories_absent_from_training(self):
        train = self.training[EXPANDED_FEATURES]
        preprocessor = make_preprocessor(EXPANDED_FEATURES, scale=True).fit(train)
        example = self.validation[EXPANDED_FEATURES].iloc[:1].assign(cbwd="unseen wind direction")
        transformed = preprocessor.transform(example)[0]
        names = preprocessor.get_feature_names_out()
        self.assertTrue(all(value == 0 for value, name in zip(transformed, names)
                            if name.startswith("categorical__cbwd_")))
        self.assertFalse(any("unseen wind direction" in name for name in names))


if __name__ == "__main__":
    unittest.main()
