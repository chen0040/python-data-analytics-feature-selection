from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC


class FeatureSelector(object):

    def __init__(self, samples, categorical_targets=None, numerical_targets=None):
        """
        Constructor method
        :param samples: X data
        :param categorical_targets: label output values which is categorical
        :param numerical_targets: numeric output values which is continuous
        """
        self.samples = samples
        self.categorical_targets = categorical_targets
        self.numerical_targets = numerical_targets
        self.pipes = list()
        self.history = dict()

    def should_remove_low_variance_features(self, p=None):
        """
        This method remove all features that have one major label over more than p% of the samples
        :param p: threshold of the percentage, if one feature has a major label over more than p% of the samples, it will
        be removed
        :return: self
        """
        if p is None:
            p = .8
        sel = VarianceThreshold(threshold=(p * (1 - p)))

        def f(sel, samples, categorical_targets, numerical_targets):
            return sel.fit_transform(samples)

        summary = 'remove all features having one major label over {}% of the samples'.format(p * 100)
        self.pipes.append((sel, f, summary))
        return self

    def should_apply_univariate_feature_selection(self, score_func=None, k=None):
        if score_func is None:
            if self.numerical_targets is not None:
                score_func = f_regression  # anova
            if self.categorical_targets is not None:
                score_func = chi2

        sel = self
        if sel.numerical_targets is not None:
            sel = sel.should_apply_univariate_feature_selection_regression(score_func, k)
        if sel.categorical_targets is not None:
            sel = sel.should_apply_univariate_feature_selection_classification(score_func, k)
        return sel

    def should_apply_univariate_feature_selection_classification(self, score_func=chi2, k=None):
        """
        This method apply chi2 test uni-variate feature selection and keep only k best features
        :param score_func:
        :param k: the number of best features to keep
        :return: self
        """

        if self.categorical_targets is None:
            return self

        if k is None:
            k = 2
        summary = 'apply classification-based univariate feature selection and keep {} best features'.format(k)
        sel = SelectKBest(score_func, k=k)

        def f(sel, samples, categorical_targets, numerical_targets):
            return sel.fit_transform(samples, categorical_targets)

        self.pipes.append((sel, f, summary))
        return self

    def should_apply_univariate_feature_selection_regression(self, score_func=f_regression, k=None):
        """
        This method apply chi2 test uni-variate feature selection and keep only k best features
        :param score_func: default is f_regression, that is anova
        :param k: the number of best features to keep
        :return: self
        """

        if self.numerical_targets is None:
            return self

        if k is None:
            k = 2
        summary = 'apply regression-based univariate feature selection and keep {} best features'.format(k)
        sel = SelectKBest(score_func, k=k)

        def f(sel, samples, categorical_targets, numerical_targets):
            return sel.fit_transform(samples, categorical_targets)

        self.pipes.append((sel, f, summary))
        return self

    def should_apply_L1_feature_selection(self, k=None):

        if k is None:
            k = 2

        sel = self
        if sel.numerical_targets is not None:
            sel = sel.should_apply_lasso(k)
        if sel.categorical_targets is not None:
            sel = sel.should_apply_linearSVC(k)

        return sel

    def should_apply_linearSVC(self, k):

        # With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features
        # selected. With Lasso, the higher the alpha parameter, the fewer features selected.

        if self.categorical_targets is None:
            return self

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)

        def f(lsvc, samples, categorical_output, numerical_output):
            X = samples
            y = categorical_output
            lsvc = lsvc.fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            return model.transform(X)

        summary = 'apply linear SVC and keep {} best features'.format(k)

        self.pipes.append((lsvc, f, summary))
        return self

    def should_apply_lasso(self, k):

        if self.numerical_targets is None:
            return self

        clf = LassoCV()

        sfm = SelectFromModel(clf, threshold=0.25)

        def f(sfm, samples, categorical_targets, numerical_targets):
            X = samples
            y = numerical_targets
            sfm.fit(X, y)
            n_features = sfm.transform(X).shape[1]

            # Reset the threshold till the number of features equals two.
            # Note that the attribute can be set directly instead of repeatedly
            # fitting the meta-transformer.
            while n_features > k:
                sfm.threshold += 0.1
                X_transform = sfm.transform(X)
                n_features = X_transform.shape[1]

            return X_transform

        summary = 'apply lasso and keep {} best features'.format(k)

        self.pipes.append((sfm, f, summary))
        return self

    def configure(self, k=None,
                  remove_low_variance_features=False,
                  apply_univariate_feature_selection=False,
                  apply_L1_feature_selection=True):
        """
        This method automatically choose the best way to apply feature selection for the samples
        :param apply_L1_feature_selection:
        :param apply_univariate_feature_selection:
        :param remove_low_variance_features: whether low variance features should be removed first
        :param k: the number of features to keep
        :return: self
        """

        if k is None:
            k = 2

        sel = self

        if remove_low_variance_features:
            sel = sel.should_remove_low_variance_features()
        if apply_univariate_feature_selection:
            sel = sel.should_apply_univariate_feature_selection(k)
        if apply_L1_feature_selection:
            sel = sel.should_apply_L1_feature_selection(k)
        return sel

    def apply(self, tracking=False):
        data = self.samples
        self.history = dict()
        for i in range(len(self.pipes)):
            pipe = self.pipes[i]
            sel, f, summary = pipe
            data1 = f(sel, samples=data, categorical_targets=self.categorical_targets,
                      numerical_targets=self.numerical_targets)
            print(summary)
            if tracking:
                self.history[summary] = data1
            data = data1

        return data
