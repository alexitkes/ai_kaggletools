"""
Unit tests for `ai_kaggletools.select_features` module.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from kaggletools.select_features import select_features_ascending
from kaggletools.select_features import select_features_descending

class TestAscending(TestCase):
    """
    Unit tests for `ai_kaggletools.select_features.select_features_ascending`
    function.
    """

    def test_normal_linear(self):
        """
        Make target variable a sum of half of normally
        distributed features. Will select_features_ascending
        find which half?
        """
        np.seed = 0
        X = pd.DataFrame(np.random.randn(1000, 6),
                         columns=["A", "B", "C", "D", "E", "F"])
        y = X["A"] + X["C"] + X["E"]
        model = LinearRegression()
        features = select_features_ascending(X, y, model)
        self.assertCountEqual(features, ["A", "C", "E"])

class TestDescending(TestCase):
    """
    Unit tests for `ai_kaggletools.select_features.select_features_descending`
    function.
    """

    def test_normal_linear(self):
        """
        Make target variable a sum of half of normally
        distributed features. Will select_features_descending
        find which half?
        """
        np.seed = 0
        X = pd.DataFrame(np.random.randn(5000, 6),
                         columns=["A", "B", "C", "D", "E", "F"])
        y = X["A"] + X["C"] + X["E"]
        y = y + np.random.randn(len(y)) * 0.01
        model = LinearRegression()
        features = select_features_descending(X, y, model)
        self.assertCountEqual(features, ["A", "C", "E"])
