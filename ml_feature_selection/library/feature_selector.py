from sklearn.feature_selection import VarianceThreshold

class FeatureSelector(object):

    def __init__(self, samples):
        self.samples = samples
        self.pipes = list()
        self.pipe_names = list()

    def remove_low_variance_features(self, p=None):
        """
        This method remove all features that have one major label over more than p% of the samples
        :param p: threshold of the percentage, if one feature has a major label over more than p% of the samples, it will
        be removed
        :return: self
        """
        if p is None:
            p = .8  
        sel = VarianceThreshold(threshold=(p * (1 - p)))
        self.pipes.append(sel)
        self.pipes.append('remove all features having one major label over {}% of the samples'.format(p * 100))
        return self

    

    def apply(self, tracking=False):
        data = self.samples
        history = dict()
        for i in range(len(self.pipes)):
            pipe = self.pipes[i]
            data1 = pipe.fit_transform(data)
            print(self.pipe_names[i])
            if tracking:
                history[self.pipe_names] = data1
            data = data1

        return history
