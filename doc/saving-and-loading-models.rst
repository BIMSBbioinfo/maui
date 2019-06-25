Saving and loading models
=========================

Maui models may be saved to disk, so that they may be loaded again at a later time.

Saving a trained model to disk
------------------------------

Saving a model to disk involves saving two files to a target directory. These files store the model weights and the maui parameters (the arguments the maui model was instantiated with). :doc:`maui` implements a the save function

.. autofunction:: maui.Maui.save

The function takes one required parameter, the destination directory where the two files will be saved. It is called directly on a maui model, like 

.. code-block:: python

    maui_model.save('/path/to/dir')


Loading a model from disk
-------------------------

Loading a model involves instantiating a new Maui instance using the parameters that were used on the model that is saved to disk, and then populating the weights of the model to the previously trained weights. Once a model is loaded, it can be used to transform new data to the latent space, or it can be trained further. :doc:`maui` has a static function to load a model from disk

.. autofunction:: maui.Maui.load

It is called directly on the Maui class, like

.. code-block:: python

    maui.Maui.load('/path/to/dir')

