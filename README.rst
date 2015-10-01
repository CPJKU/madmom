======
madmom
======

Madmom is an audio signal processing library written in Python.

The library is used internally by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

Possible acronyms are:

- Madmom Analyzes Digitized Music Of Musicians
- Mostly Audio / Dominantly Music Oriented Modules
- Madmom Analyzes Digitized Music Or Melodies
- Madmom Analyzes Digitized Music On Mushrooms

License
=======

The package has two licenses, one for source code and one for model/data files.

Source code
-----------

Unless indicated otherwise, all source code files are published under the BSD
license. For details, please see the `LICENSE <LICENSE>`_ file.

Model and data files
--------------------

Unless indicated otherwise, all model and data files are distributed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. For details,
please see the `madmom/models/LICENSE <madmom/models/LICENSE>`_ file.

If you want to include any of these files (or a variation or modification
thereof) or technology which utilises them in a commercial product, please
contact `Gerhard Widmer <http://www.cp.jku.at/people/widmer/>`_.

Installation
------------

There are several ways to install this package.

Prerequisites
-------------

To install the `madmom` package, you must have Python version 2.7 and the
following packages installed:

- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `cython <http://www.cython.org>`_
- optional:  `ffmpeg` (or `avconv` on Ubuntu Linux) if you need support for
audio files other than `.wav` with a sample rate of 44.1kHz and 16 bit depth.

Please refer to the `requirements.txt <requirements.txt>`_ file for the minimum
required versions and make sure that these modules are up to date, otherwise it
can result in unexpected errors or false computations!

Install via PyPI
----------------

The package can be installed easily via:

    pip install -r madmom

This includes the latest code and trained models.

Install via Git
---------------

If you plan to use the package as a developer, cloning or forking the Git
repository is the best option, e.g.:

    git clone https://github.com/CPJKU/madmom.git

Since the pre-trained model/data files are not included in this repository but
rather added as a Git submodule, you have either have to clone the repo
recursively:

    git clone --recursive https://github.com/CPJKU/madmom.git

or to init the submodule and fetch the data manually:

    cd /path/to/madmom
    git submodule update --init --remote

To use this package without installing it, make sure the main directory is in
your `$PYTHONPATH`, e.g. by the following command (if your are in the directory
containing this `README` file):

    export PYTHONPATH=`pwd`:$PYTHONPATH

Most modules work in Python-only mode, but some require to be compiled with
cython. To do so, please run:

    python setup.py build_ext --inplace

You can also build and install this package by running:

    python setup.py build
    python setup.py install

or:

    python setup.py develop

You might need higher privileges (use su or sudo) to install the files to a
common place like `/usr/local` or similar. Alternatively you can install the
package locally by adding the `--user` argument:

    python setup.py install --user
    python setup.py develop --user

Package structure
-----------------

The package has a very simple structure, divided into the following folders:

`/bin <bin>`_
  this folder includes example script files (i.e. executable algorithms)
`/madmom <madmom>`_
  the actual Python package
`/madmom/audio <madmom/audio>`_
  low level features (e.g. audio file handling, STFT)
`/madmom/evaluation <madmom/evaluation>`_
  evaluation code
`/madmom/features <madmom/features>`_
  higher level features (e.g. onsets, beats)
`/madmom/ml <madmom/ml>`_
  machine learning stuff (e.g. RNNs, HMMs)
`/madmom/models <madmom/models>`_
  pre-trained model/data files (see the License section)
`/madmom/test <madmom/test>`_
  tests
`/madmom/utils <madmom/utils>`_
  misc stuff (e.g. MIDI and general file handling)

Almost all low level features (i.e. everything under
`/madmom/audio <madmom/audio>`_) are divided
into a data class and a corresponding processor class. The data class refers
always to a certain instance (e.g. the STFT of an audio file), whereas the
processor classes are used to define processing chains through which the audio
is processed (i.e. most stuff in `/madmom/features <madmom/features>`_).

For usage examples please refer to the scripts in the `/bin <bin>`_ folder.

Note
----

Although we try to keep the API stable, the features are considered work in
progress and thus can change without prior notice. Do NOT expect these to stay
the same forever! If you need stable features, clone or fork this project, set
the parameters accordingly and/or pickle the processors.

Additional resources
====================

Mailing list
------------

The mailing list can be found here:
https://groups.google.com/d/forum/madmom-users

Wiki
----

The wiki can be found here: https://github.com/CPJKU/madmom/wiki

FAQ
---

Frequently asked questions can be found here:
https://github.com/CPJKU/madmom/wiki/FAQ

Contribution
------------

Please feel encouraged to contribute to this project. Every input is welcome!
