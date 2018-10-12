"""
Unit tests for the functions analyzing the Titanic data set.
"""

from unittest import TestCase

import pandas as pd
import warnings

from kaggletools.titanic import extract_title

class TestExtractTitle(TestCase):
    """
    Tests for the extract_title function.

    This class contains the following test methods.

    *   ``test_default_titles``
        Check extract_title behavior with the default title list.

    *   ``test_expand_rare_titles``
        Check extract_title behavior with the verbose title list,
        containing indexes for Dr, Military and Royal titles.

    *   ``test_invalid_titles``
        Check whether `extract_titles` function raises an exception if
        there is no 'Mr', 'Mrs', 'Miss' or 'Master' title among the
        list of allowed titles.
    """

    def setUp(self):
        """
        Create a sample data frame for testing.
        """
        self.data = pd.DataFrame({"Name": ["Smith, Mr. John",
                                           "Smith, Mrs. Jane",
                                           "Smith, Ms. Anne",
                                           "Smith, Ms. Julie",
                                           "Smith, Master. James",
                                           "Jones, Dr. Henry",
                                           "Johnson, Col. William",
                                           "Grandchester, Sir. Charles",
                                           "McCormack, Countess. Patricia"],
                                  "Sex": ["Male", "Female", "Female",
                                          "Female", "Male", "Male",
                                          "Male", "Male", "Female"],
                                  "Age": [45.0, 38.0, 15.0,
                                          12.0, 9.0, 55.0,
                                          48.0, 38.0, 35.0]})

    def test_default_titles(self):
        """
        Check extract_title behavior with the default title list.
        """
        titles = extract_title(self.data)
        # By default, the title 0 is for Mr., 1 for Mrs., 2 for Miss.,
        # 3 for Master., and 4 for all others.
        self.assertListEqual(list(titles),
                             [0, 1, 2,
                              2, 3, 4,
                              4, 4, 4])

    def test_expand_rare_titles(self):
        """
        Check extract_title behavior with the verbose title list,
        containing indexes for Dr, Military and Royal titles.
        """
        # Suppress warning message displayed every time extract_titles
        # is used with a custom title list.
        warnings.filterwarnings('ignore')
        titles = extract_title(self.data,
                               titles=["Mr", "Mrs", "Miss", "Master",
                                       "Dr", "Royal", "Military", "Rare"])
        # By default, the title 0 is for Mr., 1 for Mrs., 2 for Miss.,
        # 3 for Master., and 4 for all others.
        self.assertListEqual(list(titles),
                             [0, 1, 2,
                              2, 3, 4,
                              6, 5, 5])
        # Display any further warnings again.
        warnings.filterwarnings('default')

    def test_invalid_titles(self):
        """
        Check whether `extract_titles` function raises an exception if
        there is no 'Mr', 'Mrs', 'Miss' or 'Master' title among the
        list of allowed titles.
        """
        # Suppress warning message displayed every time extract_titles
        # is used with a custom title list.
        warnings.filterwarnings('ignore')
        with self.assertRaises(Exception):
            # Empty available title list can't be processed.
            extract_title(self.data, titles=[])
        with self.assertRaises(Exception):
            # Some optional titles given, but no required ones.
            extract_title(self.data, titles=["Rare"])
        with self.assertRaises(Exception):
            # Miss and Master titles are also required.
            extract_title(self.data, titles=["Mr", "Mrs"])
        with self.assertRaises(Exception):
            # Mr and Mrs titles are also required.
            extract_title(self.data, titles=["Master", "Miss"])
        # Display any further warnings again.
        warnings.filterwarnings('default')
