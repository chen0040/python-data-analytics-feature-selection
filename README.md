# python-data-analytics-feature-selection

Python project on feature selection

# Features

The project tries to simplify the feature selection provided by scikit-learn to a single line for the following
feature selection technique:

* Remove low-variance features
* Uni-variate feature selection technique
    * regression: f_regression (default, which is anova), 
    * classification: chi2 (default), 
* L1-based feature selection technique
    * regression: lasso
    * classification: linearSVC (default), 
    
# Usage

### For unsupervised learning

The following shows how to do feature selection with data for unsupervised learning:

```python
from ml_feature_selection.library.feature_selector import FeatureSelector
import numpy as np
X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
print(X.shape)
s = FeatureSelector(X).should_remove_low_variance_features()
X2 = s.apply()
print(X2.shape)
```

After the above steps, X2 has two features

### For classification

The following shows how to do feature selection with data for classification:

```python
from ml_feature_selection.library.feature_selector import FeatureSelector
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
s = FeatureSelector(samples=X, categorical_targets=y) \
    .should_apply_univariate_feature_selection(k=3) \
    .should_apply_L1_feature_selection(k=2) 
X2 = s.apply()
print(X2.shape)
```

### For regression

The following shows how to do feature selection with data for regression:

```python
from ml_feature_selection.library.feature_selector import FeatureSelector
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston['data'], boston['target']
print(X.shape)
s = FeatureSelector(samples=X, numerical_targets=y) \
    .should_apply_univariate_feature_selection(k=3) \
    .should_apply_L1_feature_selection(k=2) 
X2 = s.apply()
print(X2.shape)
```




