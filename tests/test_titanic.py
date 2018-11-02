"""
Unit tests for the functions analyzing the Titanic data set.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
import warnings

from kaggletools.titanic import extract_title

from kaggletools.titanic import TicketCounter
from kaggletools.titanic import CabinCounter

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

class TestTicketCounter(TestCase):
    """
    Tests the titanic.TicketCounter class.

    Tests defined here.

    *   `test_ticket_count`
        Check whether the TicketCounter object fills the TicketCount
        column properly.

    *   `test_simplified_rate`
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it in basic simplified
        way.

    *   `test_basic_rate`
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it in basic non-simplified
        way.

    *   `test_simplified_shifted_rate`
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it insimplified
        way with value 1 if anyone survived and 0 if no one
        known to survive and anyone known to die.
    """

    def setUp(self):
        """
        Create a sample data frame for testing.
        """
        self.data = pd.DataFrame({"PassengerId": [1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10],
                                  "Ticket": ["A", "B", "C", "C", "B",
                                             "C", "D", "B", "A", "E"],
                                  "Pclass": [1, 1, 2, 2, 1,
                                             2, 3, 1, 1, 3],
                                  "Survived": [1, 1, 0, 0, np.NaN,
                                               1, 0, 0, np.NaN, 1]})


    def test_ticket_count(self):
        """
        Check whether the TicketCounter object fills the TicketCount
        column properly.
        """
        TicketCounter(self.data).fill_ticket_rates()
        self.assertIn("TicketCount", self.data.columns)
        self.assertEqual(self.data.TicketCount[0], 2)
        self.assertEqual(self.data.TicketCount[1], 3)
        self.assertEqual(self.data.TicketCount[2], 3)
        self.assertEqual(self.data.TicketCount[3], 3)
        self.assertEqual(self.data.TicketCount[4], 3)
        self.assertEqual(self.data.TicketCount[5], 3)
        self.assertEqual(self.data.TicketCount[6], 1)
        self.assertEqual(self.data.TicketCount[7], 3)
        self.assertEqual(self.data.TicketCount[8], 2)
        self.assertEqual(self.data.TicketCount[9], 1)

    def test_simplified_rate(self):
        """
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it in basic simplified
        way.
        """
        TicketCounter(self.data, simplified=True).fill_ticket_rates()
        self.assertIn("TicketRate", self.data.columns)
        self.assertEqual(self.data.TicketRate[0], 0.5)
        self.assertEqual(self.data.TicketRate[1], 0.0)
        self.assertEqual(self.data.TicketRate[2], 0.5)
        self.assertEqual(self.data.TicketRate[3], 0.5)
        self.assertEqual(self.data.TicketRate[4], 0.5)
        self.assertEqual(self.data.TicketRate[5], 0.0)
        self.assertEqual(self.data.TicketRate[6], 0.5)
        self.assertEqual(self.data.TicketRate[7], 1.0)
        self.assertEqual(self.data.TicketRate[8], 1.0)
        self.assertEqual(self.data.TicketRate[9], 0.5)

    def test_basic_rate(self):
        """
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it in basic non-simplified
        way.
        """
        TicketCounter(self.data, simplified=False).fill_ticket_rates()
        self.assertIn("TicketRate", self.data.columns)
        self.assertAlmostEqual(self.data.TicketRate[0], 2.0 / 3.0)
        self.assertAlmostEqual(self.data.TicketRate[1], 0.0)
        self.assertAlmostEqual(self.data.TicketRate[2], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[3], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[4], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[5], 0.0)
        self.assertAlmostEqual(self.data.TicketRate[6], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[7], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[8], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[9], 0.5)

    def test_simplified_shifted_rate(self):
        """
        Check whether the TicketCounter object fills the TicketRate
        column properly if requested to fill it insimplified
        way with value 1 if anyone survived and 0 if no one
        known to survive and anyone known to die.
        """
        TicketCounter(self.data,
                      simplified=True,
                      fill_if_not_any_survived=True).fill_ticket_rates()
        self.assertIn("TicketRate", self.data.columns)
        self.assertAlmostEqual(self.data.TicketRate[0], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[1], 0.0)
        self.assertAlmostEqual(self.data.TicketRate[2], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[3], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[4], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[5], 0.0)
        self.assertAlmostEqual(self.data.TicketRate[6], 0.5)
        self.assertAlmostEqual(self.data.TicketRate[7], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[8], 1.0)
        self.assertAlmostEqual(self.data.TicketRate[9], 0.5)

class TestCabinCounter(TestCase):
    """
    Tests the titanic.CabinCounter class.

    Tests defined here.

    *   `test_simplified_rate`
        Check whether the CabinCounter object fills the CabinRate
        column properly if requested to fill it in basic simplified
        way.

    *   `test_basic_rate`
        Check whether the CabinCounter object fills the CabinRate
        column properly if requested to fill it in standard way.

    *   `test_filler`
        Test the filler parameter of the CabinCounter.
    """

    def setUp(self):
        """
        Create a sample data frame for testing.
        """
        self.data = pd.DataFrame({"PassengerId": [1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10],
                                  "Cabin": ["A", "B", "C", "C", "B",
                                            "C", "D", "B", "A", np.NaN],
                                  "Pclass": [1, 1, 2, 2, 1,
                                             2, 3, 1, 1, 3],
                                  "Survived": [1, 1, 0, 0, np.NaN,
                                               1, 0, 0, np.NaN, 1]})

    def test_simplified_rate(self):
        """
        Check whether the CabinCounter object fills the CabinRate
        column properly if requested to fill it in basic simplified
        way.
        """
        CabinCounter(self.data, simplified=True).fill_cabin_rates()
        self.assertIn("CabinRate", self.data.columns)
        self.assertEqual(self.data.CabinRate[0], 0.5)
        self.assertEqual(self.data.CabinRate[1], 0.0)
        self.assertEqual(self.data.CabinRate[2], 0.5)
        self.assertEqual(self.data.CabinRate[3], 0.5)
        self.assertEqual(self.data.CabinRate[4], 0.5)
        self.assertEqual(self.data.CabinRate[5], 0.0)
        self.assertEqual(self.data.CabinRate[6], 0.5)
        self.assertEqual(self.data.CabinRate[7], 1.0)
        self.assertEqual(self.data.CabinRate[8], 1.0)
        self.assertEqual(self.data.CabinRate[9], 0.5)

    def test_basic_rate(self):
        """
        Check whether the CabinCounter object fills the CabinRate
        column properly if requested to fill it in standard way.
        """
        CabinCounter(self.data, simplified=False).fill_cabin_rates()
        self.assertIn("CabinRate", self.data.columns)
        self.assertAlmostEqual(self.data.CabinRate[0], 2.0 / 3.0)
        self.assertAlmostEqual(self.data.CabinRate[1], 0.0)
        self.assertAlmostEqual(self.data.CabinRate[2], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[3], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[4], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[5], 0.0)
        self.assertAlmostEqual(self.data.CabinRate[6], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[7], 1.0)
        self.assertAlmostEqual(self.data.CabinRate[8], 1.0)
        self.assertAlmostEqual(self.data.CabinRate[9], 0.5)

    def test_filler(self):
        """
        Test the filler parameter of the CabinCounter.
        """
        filler = pd.Series(0.12345, index=self.data.index)
        CabinCounter(self.data, simplified=False, filler=filler).fill_cabin_rates()
        self.assertIn("CabinRate", self.data.columns)
        self.assertAlmostEqual(self.data.CabinRate[0], 0.12345)
        self.assertAlmostEqual(self.data.CabinRate[1], 0.0)
        self.assertAlmostEqual(self.data.CabinRate[2], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[3], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[4], 0.5)
        self.assertAlmostEqual(self.data.CabinRate[5], 0.0)
        self.assertAlmostEqual(self.data.CabinRate[6], 0.12345)
        self.assertAlmostEqual(self.data.CabinRate[7], 1.0)
        self.assertAlmostEqual(self.data.CabinRate[8], 1.0)
        self.assertAlmostEqual(self.data.CabinRate[9], 0.12345)
