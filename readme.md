<div align="center">
	<img src="hex-maui.png" alt="maui">
</div>

# maui

[![Build Status](https://travis-ci.com/BIMSBbioinfo/maui.svg?branch=master)](https://travis-ci.com/BIMSBbioinfo/maui)  [![codecov](https://codecov.io/gh/bimsbbioinfo/maui/branch/master/graph/badge.svg)](https://codecov.io/gh/bimsbbioinfo/maui) [![PyPI version](https://badge.fury.io/py/maui-tools.svg)](https://badge.fury.io/py/maui-tools) [![Documentation Status](https://readthedocs.org/projects/maui/badge/?version=latest)](https://maui.readthedocs.io/en/latest/?badge=latest)



Multi-omics Autoencoder Integration (**maui**) is a python package for multi-omics data analysis. It is based on a bayesian latent factor model, with inference done using artifiaial neural networks. For details, check out our pre-print on bioRxiv: https://www.biorxiv.org/content/early/2018/11/12/464743

## Installation

maui is only tested with Python 3.6, but probably works on 3.5+. The easiest way to install is from pypi:

	pip install maui-tools

This will install all necessary dependencies including keras an tensorflow. The default tensorflow (cpu) will be installed. If tensorflow GPU is needed, please install it prior to installation of maui.

The development version may be installed by cloning this repo and running `python setup.py install`, or, using `pip` directly from github:

	pip install -e git+https://github.com/BIMSBbioinfo/maui.git#egg=maui


#### Optional dependencies

Survival analysis functionality supplied by lifelines [1]. It may be installed directly from pip using `pip isntall lifelines`.

## Usage

See [the vignette](vignette/maui_vignette.ipynb), and check out [the documentation](https://maui.readthedocs.io/en/latest/).


## Citation

>  Evaluation of colorectal cancer subtypes and cell lines using deep learning. Jonathan Ronen, Sikander Hayat, Altuna Akalin. bioRxiv 464743; doi: https://doi.org/10.1101/464743

## Contributing

Open an issue, send us a pull request, or shoot us an e-mail.

## License

maui is released under the GPLv3 [licence](LICENSE).

---------------------
@jonathanronen, BIMSBbioinfo, 2018


[1]: https://github.com/CamDavidsonPilon/lifelines
