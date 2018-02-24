from unittest import TestCase

from ml_feature_selection.library.feature_selector import FeatureSelector
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston


class TestFeatureSelector(TestCase):
    def test_remove_low_variance_features(self):
        X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
        self.assertEqual(X.shape[1], 3)
        s = FeatureSelector(X).should_remove_low_variance_features()
        X2 = s.apply()
        self.assertEqual(X2.shape[1], 2)
        # self.assertRaises(Exception, s.demo, 2, 1, 2)

    def test_univariate_feature_selection(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        self.assertEqual(X.shape[1], 4)
        s = FeatureSelector(samples=X, categorical_targets=y).should_apply_univariate_feature_selection(k=2)
        X2 = s.apply()
        self.assertEqual(X2.shape[1], 2)

    def test_L1_feature_selection(self):
        boston = load_boston()
        X, y = boston['data'], boston['target']
        self.assertEqual(X.shape[1], 13)
        s = FeatureSelector(samples=X, numerical_targets=y).should_apply_L1_feature_selection(k=2)
        X2 = s.apply()
        self.assertEqual(X2.shape[1], 2)
