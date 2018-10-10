"""
Unit tests for the functions analyzing the Titanic data set.
"""

from unittest import TestCase

import pandas as pd

from kaggletools.titanic import extract_title

class TestExtractTitle(TestCase):
    """
    Tests for the extract_title function.
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
        titles = extract_title(self.data,
                               titles=["Mr", "Mrs", "Miss", "Master",
                                       "Dr", "Royal", "Military", "Rare"])
        # By default, the title 0 is for Mr., 1 for Mrs., 2 for Miss.,
        # 3 for Master., and 4 for all others.
        self.assertListEqual(list(titles),
                             [0, 1, 2,
                              2, 3, 4,
                              6, 5, 5])
