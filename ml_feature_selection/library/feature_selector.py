from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV


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
            score_func = 'anova'

        sel = self
        if sel.categorical_targets is not None and score_func is 'anova':
            sel = sel.should_apply_univariate_feature_selection_chi2(k)
        if sel.categorical_targets is not None and score_func is 'chi2':
            sel = sel.should_apply_univariate_feature_selection_anova(k)
        return sel

    def should_apply_univariate_feature_selection_chi2(self, k=None):
        """
        This method apply chi2 test uni-variate feature selection and keep only k best features
        :param k: the number of best features to keep
        :return: self
        """
        if k is None:
            k = 2
        summary = 'apply chi2 univariate feature selection and keep {} best features'.format(k)
        sel = SelectKBest(chi2, k=k)

        def f(sel, samples, categorical_targets, numerical_targets):
            return sel.fit_transform(samples, categorical_targets)

        self.pipes.append((sel, f, summary))
        return self

    def should_apply_univariate_feature_selection_anova(self, k=None):
        """
        This method apply chi2 test uni-variate feature selection and keep only k best features
        :param k: the number of best features to keep
        :return: self
        """
        if k is None:
            k = 2
        summary = 'apply chi2 univariate feature selection and keep {} best features'.format(k)
        sel = SelectKBest(f_regression, k=k)

        def f(sel, samples, categorical_targets, numerical_targets):
            return sel.fit_transform(samples, categorical_targets)

        self.pipes.append((sel, f, summary))
        return self

    def should_apply_lasso(self, k):
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

    def configure(self, k=None, remove_low_variance_features=False):
        """
        This method automatically choose the best way to apply feature selection for the samples
        :param remove_low_variance_features: whether low variance features should be removed first
        :param k: the number of features to keep
        :return: self
        """

        if k is None:
            k = 2

        sel = self

        if remove_low_variance_features:
            sel = sel.should_remove_low_variance_features()
        if self.categorical_targets is not None:
            sel = sel.should_apply_univariate_feature_selection(k)
        if self.numerical_targets is not None:
            sel = sel.should_apply_lasso(k)
        return sel

    def apply(self, tracking=False):
        data = self.samples
        self.history = dict()
        for i in range(len(self.pipes)):
            pipe = self.pipes[i]
            sel, f, summary = pipe
            data1 = f(sel, samples=data, categorical_targets=self.categorical_targets, numerical_targets=self.numerical_targets)
            print(summary)
            if tracking:
                self.history[summary] = data1
            data = data1

        return data
