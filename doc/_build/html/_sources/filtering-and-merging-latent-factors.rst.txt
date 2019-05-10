Filtering and Merging latent factors
====================================

We recommend running Maui with a large number of latent factors (e.g. 100), even when we expect the latent space to be of lower dimension. This way we are more likely to capture latent factors which are interesting, and the uninteresting ones can be dropped later before down-stream analysis. Maui comes with some functionality to that end.

Dropping unexplanatory latent factors
-------------------------------------

An unsupervised way to drop latent factors with low explanatory power, is to fit linear models predicting the input `x` from the latent factorz `z`. The :doc:`utils` have a function which does this. For each latent factor, a linear model is fit, predicting all input features from each latent factor. Then, the R-square is computed. Factors with an R-square score below some threshold are dropped.

.. autofunction:: maui.utils.filter_factors_by_r2

The functionality is also available directly on a trained Maui model (:doc:`maui`), which exposes a function which drops unexplanatory factors in-place:

.. autofunction:: maui.Maui.drop_unexplanatory_factors


Merging similar latent factors
------------------------------

Some times running Maui with a large number of latent factor can produce embeddings which are similar to one another. For instance, a heatmap of latent factor values may look like this:

.. _fig-colinear-factors
.. figure:: _static/colinearity.png
Heatmap of latent factors shows many latent factors are very similar.

The latent factors may be clustered and merged to produce a more succinct, even lower-dimension representation of the data, without losing much information

.. _fig-colinear-factors-merged
.. figure:: _static/colinearity-merged.png
Heatmap of latent factors after they have been merged by similarity values.

:doc:`utils` provides functionality to merge latent factors based on arbitrary distance metrics:

.. autofunction:: maui.utils.merge_factors

And functionality for the base case where factors are merged by correlation is provided in the Maui model calss:

.. autofunction:: maui.Maui.merge_similar_latent_factors


Supervised filtering of latent factors
--------------------------------------

In the case of patient data, latent factors may be assessed for usefulness based on how predictive they are of patient survival. Maui includes functionality to do this in the utilities class:

.. autofunction:: maui.utils.select_clinical_factors

For a more comprehensive example, check out `our vignette <https://github.com/BIMSBbioinfo/maui/blob/master/vignette/maui_vignette.ipynb>`_.
