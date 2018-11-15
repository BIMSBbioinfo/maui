from sklearn.base import BaseEstimator
from .autoencoders_architectures import stacked_vae

class Maui(BaseEstimator):
    """Maui (Multi-omics Autoencoder Integration) model.

    Trains a variational autoencoder to find latent factors in multi-modal data.

    Parameters
    ----------
    n_hidden: array (default [1500])
        The sizes of the hidden layers of the autoencoder architecture.
        Each element of the array specifies the number of nodes in successive
        layers of the autoencoder

    n_latent: int (default 80)
        The size of the latent layer (number of latent features)

    batch_size: int (default 100)
        The size of the mini-batches used for training the network

    epochs: int (default 400)
        The number of epoches to use for training the network
    """

    def __init__(self, n_hidden=[1500], n_latent=80, batch_size=100, epochs=400):

        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y=None, X_validation=None):
        """Train autoencoder model

        Parameters
        ----------
        X:  dict with multi-modal dataframes, containing training data, e.g.
            {'mRNA': df1, 'SNP': df2},
            df1, df2, etc. are (n_samples, n_features) pandas.DataFrame's.
            The sample names must match, the feature names need not.
        X_validation: optional, dict with multi-modal dataframes, containing validation data
            will be used to compute validation loss under training
        y:  Not used.
        """
        hist, vae, encoder, decoder = stacked_vae(
            x_train, x_test, 
            hidden_dims=self.n_hidden, latent_dim=self.n_latent,
            batch_size=self.batch_size, epochs=self.epochs)
        self.hist = hist
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        return self

    def transform(self, X):
        """Transform X into the latent space that was previously learned using
        `fit` or `fit_transform`, and return the latent factor representation.

        Parameters
        ----------
        X:  dict with multi-modal dataframes, containing training data, e.g.
            {'mRNA': df1, 'SNP': df2},
            df1, df2, etc. are (n_samples, n_features) pandas.DataFrame's.

        Returns
        -------
        z:  DataFrame (n_samples, n_latent_factors)
            Latent factors representation of the data X.
        """
        return self.encoder.predict(x)

    def fit_transform(self, X, y=None, X_validation=None):
        """Train autoencoder model, and return the latent factor representation
        of the data X.

        Parameters
        ----------
        X:  dict with multi-modal dataframes, containing training data, e.g.
            {'mRNA': df1, 'SNP': df2},
            df1, df2, etc. are (n_samples, n_features) pandas.DataFrame's.
            The sample names must match, the feature names need not.
        X_validation: optional, dict with multi-modal dataframes, containing validation data
            will be used to compute validation loss under training
        y:  Not used.
        """
        self.fit(X, X_validation=X_validation, y=y)
        return self.transform(X)


    def _validate_X(self, X):
        # TODO
        # fail if not dict
        # fail if not samples match
        # fail if some data type empty
