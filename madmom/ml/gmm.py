# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains functionality needed for fitting and scoring Gaussian
Mixture Models (GMMs) (needed e.g. in madmom.features.beats_hmm).

The needed functionality is taken from sklearn.mixture.GMM which is released
under the BSD license and was written by these authors:

- Ron Weiss <ronweiss@gmail.com>
- Fabian Pedregosa <fabian.pedregosa@inria.fr>
- Bertrand Thirion <bertrand.thirion@inria.fr>

This version works with sklearn v0.16 an onwards.
All commits until 0650d5502e01e6b4245ce99729fc8e7a71aacff3 are incorporated.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from scipy import linalg


# the following code is copied from sklearn
def logsumexp(arr, axis=0):
    """
    Computes the sum of arr assuming arr is in the log domain.

    Parameters
    ----------
    arr : numpy array
        Input data [log domain].
    axis : int, optional
        Axis to operate on.

    Returns
    -------
    numpy array
        log(sum(exp(arr))) while minimizing the possibility of over/underflow.

    Notes
    -----
    Function copied from sklearn.utils.extmath.

    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


def pinvh(a, cond=None, rcond=None, lower=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a hermetian matrix.

    Calculate a generalized inverse of a symmetric matrix using its
    eigenvalue decomposition and including all 'large' eigenvalues.

    Parameters
    ----------
    a : array, shape (N, N)
        Real symmetric or complex hermetian matrix to be pseudo-inverted.
    cond, rcond : float or None
        Cutoff for 'small' eigenvalues. Singular values smaller than rcond *
        largest_eigenvalue are considered zero.
        If None or -1, suitable machine precision is used.
    lower : boolean
        Whether the pertinent array data is taken from the lower or upper
        triangle of `a`.

    Returns
    -------
    B : array, shape (N, N)

    Raises
    ------
    LinAlgError
        If eigenvalue does not converge

    Notes
    -----
    Function copied from sklearn.utils.extmath.

    """
    a = np.asarray_chkfinite(a)
    s, u = linalg.eigh(a, lower=lower)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps

    # unlike svd case, eigh can lead to negative eigenvalues
    above_cutoff = (abs(s) > cond * np.max(abs(s)))
    psigma_diag = np.zeros_like(s)
    psigma_diag[above_cutoff] = 1.0 / s[above_cutoff]

    return np.dot(u * psigma_diag, np.conjugate(u).T)


def log_multivariate_normal_density(x, means, covars, covariance_type='diag'):
    """
    Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:

        - (n_components, n_features)             if 'spherical',
        - (n_features, n_features)               if 'tied',
        - (n_components, n_features)             if 'diag',
        - (n_components, n_features, n_features) if 'full'.

    covariance_type : {'diag', 'spherical', 'tied', 'full'}
        Type of the covariance parameters. Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in `x`
        under each of the n_components multivariate Gaussian distributions.

    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        x, means, covars)


def _log_multivariate_normal_density_diag(x, means, covars):
    """Compute Gaussian log-density at x for a diagonal model."""
    _, n_dim = x.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) +
                  np.sum((means ** 2) / covars, 1) -
                  2 * np.dot(x, (means / covars).T) +
                  np.dot(x ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(x, means, covars):
    """Compute Gaussian log-density at x for a spherical model."""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, x.shape[-1]))
    return _log_multivariate_normal_density_diag(x, means, cv)


def _log_multivariate_normal_density_tied(x, means, covars):
    """Compute Gaussian log-density at X for a tied model"""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(x, means, cv)


def _log_multivariate_normal_density_full(x, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = x.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (x - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


# provide a basic GMM implementation which has the score() method implemented
class GMM(object):
    """
    Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.


    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.
    covariance_type : {'diag', 'spherical', 'tied', 'full'}
        String describing the type of covariance parameters to
        use. Defaults to 'diag'.

    Attributes
    ----------
    `weights_` : array, shape (n_components,)
        This attribute stores the mixing weights for each mixture component.
    `means_` : array, shape (n_components, n_features)
        Mean parameters for each mixture component.
    `covars_` : array
        Covariance parameters for each mixture component. The shape
        depends on `covariance_type`.::

        - (n_components, n_features)             if 'spherical',
        - (n_features, n_features)               if 'tied',
        - (n_components, n_features)             if 'diag',
        - (n_components, n_features, n_features) if 'full'.

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    """

    def __init__(self, n_components=1, covariance_type='full'):

        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)
        # save parameters
        self.n_components = n_components
        self.covariance_type = covariance_type
        # init variables
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = None
        self.covars_ = None
        self.converged_ = False

    def score_samples(self, x):
        """
        Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds
            to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in `x`.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.

        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if x.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of x is not compatible with self')

        lpr = (log_multivariate_normal_density(x, self.means_, self.covars_,
                                               self.covariance_type) +
               np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, x):
        """
        Compute the log probability under the model.

        Parameters
        ----------
        x : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in `x`.

        """
        logprob, _ = self.score_samples(x)
        return logprob

    def fit(self, x, random_state=None, tol=1e-3, min_covar=1e-3,
            n_iter=100, n_init=1, params='wmc', init_params='wmc'):
        """
        Estimate model parameters with the expectation-maximization algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        x : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row corresponds
            to a single data point.
        random_state: RandomState or an int seed (0 by default)
            A random number generator instance.
        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting.
        tol : float, optional
            Convergence threshold. EM iterations will stop when average
            gain in log-likelihood is below this threshold.
        n_iter : int, optional
            Number of EM iterations to perform.
        n_init : int, optional
            Number of initializations to perform, the best results is kept.
        params : string, optional
            Controls which parameters are updated in the training process.
            Can contain any combination of 'w' for weights, 'm' for means,
            and 'c' for covars.
        init_params : string, optional
            Controls which parameters are updated in the initialization
            process.  Can contain any combination of 'w' for weights,
            'm' for means, and 'c' for covars.

        """
        import sklearn.mixture
        # first initialise a sklearn.mixture.GMM object
        gmm = sklearn.mixture.GMM(n_components=self.n_components,
                                  covariance_type=self.covariance_type,
                                  random_state=random_state, tol=tol,
                                  min_covar=min_covar, n_iter=n_iter,
                                  n_init=n_init, params=params,
                                  init_params=init_params)
        # fit this GMM
        gmm.fit(x)
        # copy the needed information
        self.converged_ = gmm.converged_
        self.covars_ = gmm.covars_
        self.means_ = gmm.means_
        self.weights_ = gmm.weights_
        # and return self
        return self
