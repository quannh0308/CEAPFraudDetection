"""
Feature engineering module for ML experimentation workflow.

This module provides the FeatureEngineer class for creating derived features,
selecting important features, and analyzing feature importance for fraud
detection model experiments.

Example:
    from feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()

    # Create new features
    df = engineer.create_time_features(df)
    df = engineer.create_amount_features(df)
    df = engineer.create_interaction_features(df, [('V1', 'V2'), ('V3', 'V4')])

    # Select top features
    selected, scores = engineer.select_features_univariate(X, y, k=20)

    # Analyze importance
    importance_df = engineer.analyze_feature_importance(X, y)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif


class FeatureEngineer:
    """
    Utilities for feature engineering experiments.

    Provides methods for creating derived features (time-based, amount-based,
    interaction) and selecting/ranking features using statistical tests,
    recursive elimination, and importance analysis.

    Example:
        engineer = FeatureEngineer()
        df = engineer.create_time_features(df)
        selected, scores = engineer.select_features_univariate(X, y, k=10)
    """

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from transaction timestamp.

        Derives hour, day of week, weekend indicator, and night indicator
        from the ``Time`` column (assumed to be seconds since epoch).

        Args:
            df: DataFrame containing a ``Time`` column in seconds.

        Returns:
            DataFrame with added columns: ``hour``, ``day_of_week``,
            ``is_weekend``, ``is_night``.

        Example:
            df = engineer.create_time_features(df)
        """
        timestamps = pd.to_datetime(df["Time"], unit="s")
        df["hour"] = timestamps.dt.hour
        df["day_of_week"] = timestamps.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].between(22, 6).astype(int)
        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features from transaction amount.

        Derives log-transformed, squared, and square-root versions of the
        ``Amount`` column.

        Args:
            df: DataFrame containing an ``Amount`` column.

        Returns:
            DataFrame with added columns: ``amount_log``, ``amount_squared``,
            ``amount_sqrt``.

        Example:
            df = engineer.create_amount_features(df)
        """
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_squared"] = df["Amount"] ** 2
        df["amount_sqrt"] = np.sqrt(df["Amount"])
        return df

    def create_interaction_features(
        self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified column pairs.

        For each pair ``(feat1, feat2)``, a new column ``feat1_x_feat2`` is
        created containing the element-wise product.

        Args:
            df: Source DataFrame.
            feature_pairs: List of ``(column_a, column_b)`` tuples.

        Returns:
            DataFrame with added interaction columns.

        Example:
            df = engineer.create_interaction_features(df, [('V1', 'V2')])
        """
        for feat1, feat2 in feature_pairs:
            df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
        return df

    def select_features_univariate(
        self, X: pd.DataFrame, y: Any, k: int = 20
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select top *k* features using univariate statistical tests.

        Uses ``SelectKBest`` with the ANOVA F-value (``f_classif``) scorer.

        Args:
            X: Feature DataFrame.
            y: Target array.
            k: Number of top features to select.

        Returns:
            Tuple of (selected feature names, scores DataFrame sorted
            descending by score).

        Example:
            selected, scores = engineer.select_features_univariate(X, y, k=10)
        """
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)

        scores = pd.DataFrame(
            {"feature": X.columns, "score": selector.scores_}
        ).sort_values("score", ascending=False)

        selected_features = scores.head(k)["feature"].tolist()
        return selected_features, scores

    def select_features_rfe(
        self, X: pd.DataFrame, y: Any, n_features: int = 20
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features using Recursive Feature Elimination (RFE).

        Uses a ``RandomForestClassifier`` as the underlying estimator.

        Args:
            X: Feature DataFrame.
            y: Target array.
            n_features: Number of features to select.

        Returns:
            Tuple of (selected feature names, ranking DataFrame sorted
            ascending by rank).

        Example:
            selected, ranking = engineer.select_features_rfe(X, y, n_features=15)
        """
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)

        selected_features = X.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame(
            {"feature": X.columns, "ranking": selector.ranking_}
        ).sort_values("ranking")

        return selected_features, feature_ranking

    def analyze_feature_importance(
        self, X: pd.DataFrame, y: Any
    ) -> pd.DataFrame:
        """
        Analyze feature importance using a Random Forest classifier.

        Args:
            X: Feature DataFrame.
            y: Target array.

        Returns:
            DataFrame with ``feature`` and ``importance`` columns, sorted
            descending by importance.

        Example:
            importance_df = engineer.analyze_feature_importance(X, y)
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df
