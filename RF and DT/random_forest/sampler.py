import numpy as np
#from numpy.conftest import na_object

#from decision_tree.tests.my_code import max_samples


class BaseSampler:
    def __init__(self, max_samples=1, bootstrap=False, random_state=None):
        """
        Parameters
        ----------
        bootstrap : Boolean
            if True then use bootstrap sampling
        max_samples : float in [0;1]
            proportion of sampled examples
        """
        self.random_state = np.random.RandomState(random_state)
        self.bootstrap = bootstrap
        self.max_samples = max_samples

    def sample_indices(self, n_objects):
        """
        Parameters
        ----------
        n_objects : int > 0
            number of sampling objects
        """
        #if self.random_state is not None:
        rng = np.random.default_rng(self.random_state)
        n_samples = self.max_samples * n_objects
        if self.bootstrap:
            sampled_indices = rng.choice(n_objects, size=n_samples, replace=True)
        else:
            sampled_indices = rng.choice(n_objects, size=n_samples, replace=False)
        print(sampled_indices)
        return sampled_indices

    def sample(self, x, y=None):
        # abstract method
        raise NotImplementedError


class ObjectSampler(BaseSampler):
    def __init__(self, max_samples=1, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled, y_sampled : numpy ndarrays of shape (n_samples, n_features) and (n_samples,)
        """

        n_objects = x.shape[0]

        sampled_indices = self.sample_indices(n_objects)

        x_sampled = x[sampled_indices]
        y_sampled = y[sampled_indices] if y is not None else None

        return x_sampled, y_sampled


class FeatureSampler(BaseSampler):
    def __init__(self, max_samples=1, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled : numpy ndarrays of shape (n_objects, n_features_sampled)
        """
        n_objects = x.shape[1]

        sampled_indices = self.sample_indices(n_objects)

        x_sampled = x[sampled_indices]
        y_sampled = y[sampled_indices] if y is not None else None

        return x_sampled, y_sampled