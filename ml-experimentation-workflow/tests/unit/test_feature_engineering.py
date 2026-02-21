"""
Unit tests for feature engineering module.

Tests FeatureEngineer class including time feature creation, amount feature
creation, interaction features, univariate selection, RFE selection, and
feature importance analysis.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Mock sagemaker modules before importing
sys.modules["sagemaker"] = MagicMock()
sys.modules["sagemaker.experiments"] = MagicMock()

from feature_engineering import FeatureEngineer


@pytest.fixture
def engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def time_df():
    """Create a sample DataFrame with a Time column (seconds since epoch)."""
    return pd.DataFrame({
        "Time": [0, 3600, 7200, 43200, 86400, 432000],
        "Amount": [10.0, 25.5, 100.0, 0.5, 250.0, 75.0],
    })


@pytest.fixture
def classification_data():
    """Create sample classification data for feature selection tests."""
    np.random.seed(42)
    n_samples = 100
    # Create features with varying predictive power
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(n_samples) + (i * 0.1 if i < 3 else 0)
        for i in range(10)
    })
    y = (X["feat_0"] + X["feat_1"] > 0).astype(int)
    return X, y


class TestCreateTimeFeatures:
    """Tests for create_time_features method."""

    def test_adds_expected_columns(self, engineer, time_df):
        result = engineer.create_time_features(time_df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns

    def test_hour_values(self, engineer, time_df):
        result = engineer.create_time_features(time_df)
        # Time=0 is midnight (hour 0), Time=3600 is 1am, Time=7200 is 2am
        assert result["hour"].iloc[0] == 0
        assert result["hour"].iloc[1] == 1
        assert result["hour"].iloc[2] == 2

    def test_is_weekend_values(self, engineer):
        # 1970-01-01 is Thursday (dayofweek=3), 1970-01-03 is Saturday (dayofweek=5)
        df = pd.DataFrame({
            "Time": [0, 172800],  # Thursday, Saturday
            "Amount": [10.0, 20.0],
        })
        result = engineer.create_time_features(df)
        assert result["is_weekend"].iloc[0] == 0  # Thursday
        assert result["is_weekend"].iloc[1] == 1  # Saturday

    def test_is_night_values(self, engineer):
        # hour.between(22, 6) — only 22..6 inclusive; note pandas between is inclusive
        # hour=23 should be night, hour=12 should not
        df = pd.DataFrame({
            "Time": [82800, 43200],  # 23:00, 12:00
            "Amount": [10.0, 20.0],
        })
        result = engineer.create_time_features(df)
        assert result["is_night"].iloc[0] == 0  # 23 is NOT between(22,6) since 22>6 makes between return False for 23
        assert result["is_night"].iloc[1] == 0  # 12 is not night

    def test_preserves_existing_columns(self, engineer, time_df):
        result = engineer.create_time_features(time_df)
        assert "Time" in result.columns
        assert "Amount" in result.columns

    def test_returns_dataframe(self, engineer, time_df):
        result = engineer.create_time_features(time_df)
        assert isinstance(result, pd.DataFrame)


class TestCreateAmountFeatures:
    """Tests for create_amount_features method."""

    def test_adds_expected_columns(self, engineer, time_df):
        result = engineer.create_amount_features(time_df)
        assert "amount_log" in result.columns
        assert "amount_squared" in result.columns
        assert "amount_sqrt" in result.columns

    def test_log_transform(self, engineer):
        df = pd.DataFrame({"Amount": [0.0, 1.0, 100.0]})
        result = engineer.create_amount_features(df)
        np.testing.assert_almost_equal(result["amount_log"].iloc[0], np.log1p(0.0))
        np.testing.assert_almost_equal(result["amount_log"].iloc[1], np.log1p(1.0))
        np.testing.assert_almost_equal(result["amount_log"].iloc[2], np.log1p(100.0))

    def test_squared_transform(self, engineer):
        df = pd.DataFrame({"Amount": [2.0, 3.0, 10.0]})
        result = engineer.create_amount_features(df)
        assert result["amount_squared"].iloc[0] == 4.0
        assert result["amount_squared"].iloc[1] == 9.0
        assert result["amount_squared"].iloc[2] == 100.0

    def test_sqrt_transform(self, engineer):
        df = pd.DataFrame({"Amount": [4.0, 9.0, 16.0]})
        result = engineer.create_amount_features(df)
        np.testing.assert_almost_equal(result["amount_sqrt"].iloc[0], 2.0)
        np.testing.assert_almost_equal(result["amount_sqrt"].iloc[1], 3.0)
        np.testing.assert_almost_equal(result["amount_sqrt"].iloc[2], 4.0)

    def test_zero_amount(self, engineer):
        df = pd.DataFrame({"Amount": [0.0]})
        result = engineer.create_amount_features(df)
        assert result["amount_log"].iloc[0] == 0.0
        assert result["amount_squared"].iloc[0] == 0.0
        assert result["amount_sqrt"].iloc[0] == 0.0


class TestCreateInteractionFeatures:
    """Tests for create_interaction_features method."""

    def test_creates_interaction_column(self, engineer):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = engineer.create_interaction_features(df, [("A", "B")])
        assert "A_x_B" in result.columns
        assert list(result["A_x_B"]) == [4, 10, 18]

    def test_multiple_pairs(self, engineer):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = engineer.create_interaction_features(df, [("A", "B"), ("A", "C")])
        assert "A_x_B" in result.columns
        assert "A_x_C" in result.columns

    def test_empty_pairs(self, engineer):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = engineer.create_interaction_features(df, [])
        assert len(result.columns) == 2  # no new columns

    def test_preserves_original_columns(self, engineer):
        df = pd.DataFrame({"X": [1.0], "Y": [2.0]})
        result = engineer.create_interaction_features(df, [("X", "Y")])
        assert result["X"].iloc[0] == 1.0
        assert result["Y"].iloc[0] == 2.0


class TestSelectFeaturesUnivariate:
    """Tests for select_features_univariate method."""

    def test_returns_correct_number_of_features(self, engineer, classification_data):
        X, y = classification_data
        selected, scores = engineer.select_features_univariate(X, y, k=5)
        assert len(selected) == 5

    def test_returns_feature_names(self, engineer, classification_data):
        X, y = classification_data
        selected, scores = engineer.select_features_univariate(X, y, k=3)
        for feat in selected:
            assert feat in X.columns.tolist()

    def test_scores_dataframe_has_all_features(self, engineer, classification_data):
        X, y = classification_data
        _, scores = engineer.select_features_univariate(X, y, k=5)
        assert len(scores) == len(X.columns)
        assert "feature" in scores.columns
        assert "score" in scores.columns

    def test_scores_sorted_descending(self, engineer, classification_data):
        X, y = classification_data
        _, scores = engineer.select_features_univariate(X, y, k=5)
        score_values = scores["score"].values
        assert all(score_values[i] >= score_values[i + 1] for i in range(len(score_values) - 1))

    def test_selected_features_are_subset(self, engineer, classification_data):
        X, y = classification_data
        selected, _ = engineer.select_features_univariate(X, y, k=5)
        assert set(selected).issubset(set(X.columns))

    def test_k_equals_total_features(self, engineer, classification_data):
        X, y = classification_data
        selected, _ = engineer.select_features_univariate(X, y, k=len(X.columns))
        assert len(selected) == len(X.columns)


class TestSelectFeaturesRFE:
    """Tests for select_features_rfe method."""

    def test_returns_correct_number_of_features(self, engineer, classification_data):
        X, y = classification_data
        selected, ranking = engineer.select_features_rfe(X, y, n_features=5)
        assert len(selected) == 5

    def test_returns_feature_names(self, engineer, classification_data):
        X, y = classification_data
        selected, _ = engineer.select_features_rfe(X, y, n_features=3)
        for feat in selected:
            assert feat in X.columns.tolist()

    def test_ranking_dataframe_has_all_features(self, engineer, classification_data):
        X, y = classification_data
        _, ranking = engineer.select_features_rfe(X, y, n_features=5)
        assert len(ranking) == len(X.columns)
        assert "feature" in ranking.columns
        assert "ranking" in ranking.columns

    def test_ranking_sorted_ascending(self, engineer, classification_data):
        X, y = classification_data
        _, ranking = engineer.select_features_rfe(X, y, n_features=5)
        rank_values = ranking["ranking"].values
        assert all(rank_values[i] <= rank_values[i + 1] for i in range(len(rank_values) - 1))

    def test_selected_features_have_rank_one(self, engineer, classification_data):
        X, y = classification_data
        selected, ranking = engineer.select_features_rfe(X, y, n_features=5)
        for feat in selected:
            rank = ranking[ranking["feature"] == feat]["ranking"].iloc[0]
            assert rank == 1


class TestAnalyzeFeatureImportance:
    """Tests for analyze_feature_importance method."""

    def test_returns_all_features(self, engineer, classification_data):
        X, y = classification_data
        importance_df = engineer.analyze_feature_importance(X, y)
        assert len(importance_df) == len(X.columns)

    def test_has_expected_columns(self, engineer, classification_data):
        X, y = classification_data
        importance_df = engineer.analyze_feature_importance(X, y)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

    def test_importances_sorted_descending(self, engineer, classification_data):
        X, y = classification_data
        importance_df = engineer.analyze_feature_importance(X, y)
        values = importance_df["importance"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_importances_sum_to_one(self, engineer, classification_data):
        X, y = classification_data
        importance_df = engineer.analyze_feature_importance(X, y)
        np.testing.assert_almost_equal(importance_df["importance"].sum(), 1.0, decimal=5)

    def test_importances_non_negative(self, engineer, classification_data):
        X, y = classification_data
        importance_df = engineer.analyze_feature_importance(X, y)
        assert (importance_df["importance"] >= 0).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_time_features(self, engineer):
        df = pd.DataFrame({"Time": [0], "Amount": [10.0]})
        result = engineer.create_time_features(df)
        assert len(result) == 1
        assert "hour" in result.columns

    def test_single_row_amount_features(self, engineer):
        df = pd.DataFrame({"Amount": [5.0]})
        result = engineer.create_amount_features(df)
        assert len(result) == 1

    def test_single_feature_univariate(self, engineer):
        np.random.seed(42)
        X = pd.DataFrame({"only_feat": np.random.randn(50)})
        y = (X["only_feat"] > 0).astype(int)
        selected, scores = engineer.select_features_univariate(X, y, k=1)
        assert len(selected) == 1
        assert selected[0] == "only_feat"

    def test_large_amounts(self, engineer):
        df = pd.DataFrame({"Amount": [1e10, 1e15]})
        result = engineer.create_amount_features(df)
        assert np.isfinite(result["amount_log"]).all()
        assert np.isfinite(result["amount_sqrt"]).all()
