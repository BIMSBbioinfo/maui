Multi-modal Autoencoders
========================

Autoencoders are hourglass-shaped neural networks that are trained to reconstruct the input data after passing it through a bottleneck layer. Thereby, autoencoders learn an efficient lower dimension representation of high-dimensional data, a "latent factor" representation of the data. The  Multi-modal autoencoders take data from different modalities and learn latent factor representations of the data. In the figure below, the direction of data flow is left to right. The three different-colored matrices on the left represent data from three different modalities. The three matrices on the right represent the reconstruction of the input data, and the mixed-color matrix on the bottom represents the latent factor view of the data.

.. _fig-integration-autoencoder
.. figure:: _static/integration-autoencoder.png
Integration Autoencoder


Variational Autoencoders
------------------------

Maui uses a Variational Autoencoder, which means it learns a bayesian latent variable model. This is achieved by minimizing the following loss function:

:math:`\mathcal{L} = -\mathbf{E}_{q(z|x)}\big[ log(p(x|z)) \big] + D_{KL}\big( q(z|x)~\|~p(z) \big)`

The first term represents the cross-entropy reconstruction loss, and the second term is the Kullback-Leibler divergence between the latent factors distribution and a gaussian prior :math:`p(z)`.


Stacked Autoencoders
--------------------

As the figure above indicates, it is possible to insert extra layers between the input and the bottleneck layers, and between the bottleneck and the output layers. This is sometimes called stacked autoencoders. :doc:`maui` allows this architecture to be varied when instantiating the model, using the ``n_hidden`` parameter. The ``n_latent`` parameter determines the size of the bottleneck layer (latent factor layer).

.. code-block:: python

    maui_model = maui.Maui(n_hidden=[900], n_latent=70)

instantiates a Maui model with one hidden layer with 900 units, and 70 units in the bottleneck layer, while 

.. code-block:: python

    maui_model = maui.Maui(n_hidden=[1300, 900], n_latent=60)

will instantiate a maui model with two hidden layers with 1300 and 900 units, respectively.