"""
Unit tests for `ai_kaggletools.select_features` module.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from kaggletools.select_features import select_features_ascending
from kaggletools.select_features import select_features_descending
from kaggletools.select_features import SumDiffTransformer

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

    def test_normal_linear_with_starting_features(self):
        """
        Make target variable a sum of half of normally
        distributed features. Will select_features_ascending
        find which half if starts with one garbage feature
        it can't refuse?
        """
        np.seed = 0
        X = pd.DataFrame(np.random.randn(1000, 6),
                         columns=["A", "B", "C", "D", "E", "F"])
        y = X["A"] + X["C"] + X["E"]
        model = LinearRegression()
        features = select_features_ascending(X, y, model, starting_features="D")
        self.assertCountEqual(features, ["A", "C", "D", "E"])

    def test_normal_linear_with_njobs(self):
        """
        Make target variable a sum of half of normally
        distributed features. Will select_features_ascending
        find which half? And will it do it faster if `n_jobs`
        parameter given?
        """
        np.seed = 0
        X = pd.DataFrame(np.random.randn(1000, 6),
                         columns=["A", "B", "C", "D", "E", "F"])
        y = X["A"] + X["C"] + X["E"]
        model = LinearRegression()
        features = select_features_ascending(X, y, model, n_jobs=-1)
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
        y = y + np.random.randn(len(y)) * 0.005
        model = LinearRegression()
        features = select_features_descending(X, y, model)
        self.assertCountEqual(features, ["A", "C", "E"])

class TestSumDiff(TestCase):
    """
    Tests for the `SumDiffTransformer` class.
    """

    def test_simple(self):
        """
        Just check whether the SumDiffTransformer class does what it
        should.
        """
        # Every feature is a column...
        X = np.array([[1, 2, 3], [4, 5, 6]]).T
        X = SumDiffTransformer().fit_transform(X)
        # AssertEqual causes an error for numpy arrays.
        X_expected = np.array([[1, 2, 3],
                               [-3, -3, -3],
                               [5, 7, 9],
                               [4, 5, 6]]).T
        self.assertAlmostEqual((X - X_expected).min(), 0.0)
        self.assertAlmostEqual((X - X_expected).max(), 0.0)
