ChannelAttribution
======================

Advertisers use a variety of online marketing channels to reach consumers and they want to know the degree each channel contributes to their marketing success. This is called online multi-channel attribution problem. This package contains a probabilistic algorithm for the attribution problem. The model uses a k-order Markov representation to identify structural correlations in the customer journey data. The package also contains three heuristic algorithms (first-touch, last-touch and linear-touch approach) for the same problem. The algorithms are implemented in C++. 

Installation
------------

### From PyPi

```bash
pip install --upgrade setuptools
pip install ChannelAttribution
```


Generating distribution archives
--------------------------------

```bash
python setup.py sdist bdist_wheel
```

Generating documentation
------------------------

```bash
pip install Sphinx
pip install rinohtype

mkdir /src/cypack/docs
cd /src/cypack/docs

sphinx-quickstart

make clean && make html
sphinx-build -b rinoh . _build/rinoh
```
