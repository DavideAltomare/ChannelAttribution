*ChannelAttribution*
====================

Advertisers use a variety of online marketing channels to reach consumers and they want to know the degree each channel contributes to their marketing success. This is called online multi-channel attribution problem. *ChannelAttribution* contains a probabilistic algorithm for the attribution problem. The model uses a k-order Markov representation to identify structural correlations in the customer journey data. The library also contains three heuristic algorithms (first-touch, last-touch and linear-touch approach) for the same problem. The algorithms are implemented in C++. 

Python library
--------------

/python contains Python library source code


R package
---------

/R contains R package source code

website
-------

/website contains website source code


Python installation
-------------------

Note! Only Python3 is supported!

### From PyPi

```bash
pip install --upgrade setuptools
pip install Cython
pip install ChannelAttribution
```

Generating Python documentation
-------------------------------

```bash
cd ...python\src\cypack

python generate_doc.py
```

The following .pdf will be generated:

```bash
...python/src/cypack/docs/_build/rinoh/channelattribution.pdf
```