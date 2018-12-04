Data and Normalization
======================

All features need to be scaled and centered prior to feeding to the neural network of maui. This limitation comes from the fact that the last layer, where the input data is reconstructed, uses Sigmoid activations. Maui uses Batch Normalization during training, where each input feature is normalized at each minibatch. We still recommend that data be scaled prior to training. We provide a function that does this in the :doc:`utils`.

.. autofunction:: maui.utils.scale
