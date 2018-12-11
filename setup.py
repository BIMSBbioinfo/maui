from setuptools import setup

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(name='maui-tools',
    version='0.1.1',
    description="Multi-omics Autoencoder Integration",
    author='Jonathan Ronen',
    license='GPLv3',
    author_email='yablee@gmail.com',
    url='https://github.com/BIMSBbioinfo/maui',
    packages=['maui'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pytest',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'keras',
        'tensorflow'
    ],
)
