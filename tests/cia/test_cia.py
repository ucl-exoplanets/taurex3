
import unittest
from unittest.mock import patch, mock_open
from taurex.cia.cia import CIA
from taurex.cia.hitrancia import HitranCIA
from taurex.cia.picklecia import PickleCIA
import numpy as np

pickle_test_data = {'xsecarr': np.array([[2.668e-57, 2.931e-57, 3.203e-57, 1.226e-59, 1.220e-59,
                                          1.213e-59],
                                         [2.379e-57, 2.613e-57, 2.856e-57, 1.654e-59, 1.646e-59,
                                          1.636e-59],
                                         [2.137e-57, 2.348e-57, 2.568e-57, 2.164e-59, 2.153e-59,
                                          2.142e-59]]),
                    'wno': np.array([20,    21,    22, 9998,  9999, 10000], dtype=np.int64),
                    'comments': ['H2-H2 collision induced absorption',
                                 '200-3000K in steps of 25K',
                                 'C. Richard, I.E. Gordon, L.S. Rothman, M. Abel, L. Frommhold, M. Gustafsson, et al, JQSRT 113, 1276-1285 (2012).',
                                 'Created with TauREx at GMT 2016-01-07 17:21:11'],
                    't': np.array([200.,  225.,  250.])}


class CIATest(unittest.TestCase):

    def test_notimplemented(self):
        op = CIA('My name', 'HEH-2HF')
        with self.assertRaises(NotImplementedError):
            op.cia(100)
        self.assertEqual(op.pairName, 'HEH-2HF')
        self.assertEqual(op.pairOne, 'HEH')
        self.assertEqual(op.pairTwo, '2HF')


class PickleCIATest(unittest.TestCase):

    def setUp(self):
        import pickle
        self.data = pickle.dumps(pickle_test_data)
        self.pop = None
        with patch("builtins.open", mock_open(read_data=self.data)) as mock_file:
            self.pop = PickleCIA('/unittestfile/test.db', 'HeH-2HF')

    def test_properties(self):

        self.assertEqual(self.pop.pairName, 'HeH-2HF')
        np.testing.assert_equal(
            pickle_test_data['t'], self.pop.temperatureGrid)
        np.testing.assert_equal(
            pickle_test_data['wno'], self.pop.wavenumberGrid)
        np.testing.assert_equal(
            pickle_test_data['xsecarr'], self.pop._xsec_grid)

    def test_pathname_naming(self):
        with patch("builtins.open", mock_open(read_data=self.data)) as mock_file:
            pop = PickleCIA('/unittestfile/HeH-2HF.db')
            mock_file.assert_called_once_with('/unittestfile/HeH-2HF.db', 'rb')
            self.assertEqual(pop.pairName, 'HeH-2HF')

    def test_cia_calc(self):
        np.testing.assert_allclose(self.pop.cia(
            225), pickle_test_data['xsecarr'][1])

    def test_max_min_temps(self):
        np.testing.assert_equal(self.pop.cia(
            100000000), self.pop._xsec_grid[-1])
        np.testing.assert_equal(self.pop.cia(
            0.0000001), self.pop._xsec_grid[0])


class HitranCIATest(unittest.TestCase):

    def test_reads_problematic_headers(self):
        test_cases = [
            (
                (
                    "eq-H2 -- eq-H2\n"
                    "H2-H2 20 20 1 200 1.0\n"
                    "20 2.668e-57\n"
                ),
                np.array([200.0]),
                np.array([20.0]),
            ),
            (
                (
                    "eq-H2 -- eq-H2 0.250000 2400.0000 8 40.000 3.683e-44 -.999 34\n"
                    "0.2500 4.706E-51\n"
                    "0.5000 1.957E-50\n"
                    "0.7500 4.401E-50\n"
                    "1.0000 7.983E-50\n"
                    "1.2500 1.399E-49\n"
                    "1.5000 1.878E-49\n"
                    "1.7500 2.471E-49\n"
                    "2.0000 3.407E-49\n"
                ),
                np.array([40.0]),
                np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            ),
        ]

        for hitran_data, temperature_grid, wavenumber_grid in test_cases:
            with self.subTest(header=hitran_data.splitlines()[0]):
                with patch("builtins.open", mock_open(read_data=hitran_data)):
                    cia = HitranCIA('/unittestfile/H2-H2.cia')

                self.assertEqual(cia.pairName, 'H2-H2')
                np.testing.assert_equal(cia.temperatureGrid, temperature_grid)
                np.testing.assert_equal(cia.wavenumberGrid, wavenumber_grid)
