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

It includes reference implementations for some music information retrieval
algorithms, please see the `References`_ section.

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
`Creative Commons Attribution-NonCommercial-ShareAlike 4.0
<http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_ license.

If you want to include any of these files (or a variation or modification
thereof) or technology which utilises them in a commercial product, please
contact `Gerhard Widmer <http://www.cp.jku.at/people/widmer/>`_.

Installation
============

Prerequisites
-------------

To install the ``madmom`` package, you must have Python version 2.7 and the
following packages installed:

- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `cython <http://www.cython.org>`_

If you need support for audio files other than ``.wav`` with a sample rate of
44.1kHz and 16 bit depth, you need ``ffmpeg`` (or ``avconv`` on Ubuntu Linux).

Please refer to the `requirements.txt <requirements.txt>`_ file for the minimum
required versions and make sure that these modules are up to date, otherwise it
can result in unexpected errors or false computations!

Install from package
--------------------

The easiest way to install the package is via ``pip`` from the `PyPI (Python
Package Index) <https://pypi.python.org/pypi>`_:

    pip install madmom

This includes the latest code and trained models and will install all
dependencies automatically. It will also install the executable scripts to a
common place (e.g. ``/usr/local/bin``) which should be in your ``$PATH``
already. ``pip`` will output the install location.

You might need higher privileges (use su or sudo) to install the package, model
files and scripts globally. Alternatively you can install the package locally
(i.e. only for you) by adding the ``--user`` argument:

    pip install --user madmom

Depending on your platform, the scripts will be copied to a folder which
might not be included in your ``$PATH`` (e.g. ``~/Library/Python/2.7/bin``
on Mac OS X or ``~/.local/bin`` on Ubuntu Linux), so please call the scripts
directly or add their path to your ``$PATH`` environment variable:

    export PATH='path/to/scripts':$PATH

Install from source
-------------------

If you plan to use the package as a developer, cloning the Git repository is
the best option, e.g.:

    git clone https://github.com/CPJKU/madmom.git

Since the pre-trained model/data files are not included in this repository but
rather added as a Git submodule, you either have to clone the repo recursively:

    git clone --recursive https://github.com/CPJKU/madmom.git

or init the submodule and fetch the data manually:

    cd /path/to/madmom

    git submodule update --init --remote

You can then build and install this package by running:

    python setup.py build

    python setup.py install

You might need higher privileges (use su or sudo) to install the files to a
common place like ``/usr/local`` or similar. Alternatively you can install the
package locally by adding the ``--user`` argument:

    python setup.py install --user

Install for development
-----------------------

If you want to actively work on the package, please follow the git instructions
from the `Install from source`_ section.

You can then either include the package directory in your ``$PYTHONPATH``,
e.g. by the following command (if your are in the directory containing this
``README`` file):

    export PYTHONPATH=`pwd`:$PYTHONPATH

or you can install the package in development mode:

    python setup.py develop

If you are not using the development variant or if you change any ``.pyx`` or
``.pxd`` files, you have to (re-)compile some modules with Cython. To do so,
please run:

    python setup.py build_ext --inplace

Again, you can install the package locally by adding the ``--user`` argument:

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
`/madmom/utils <madmom/utils>`_
  misc stuff (e.g. MIDI and general file handling)
`/tests <tests>`_
  tests

Almost all low level features (i.e. everything under
`/madmom/audio <madmom/audio>`_) are divided
into a data class and a corresponding processor class. The data class refers
always to a certain instance (e.g. the STFT of an audio file), whereas the
processor classes are used to define processing chains through which the audio
is processed (i.e. most stuff in `/madmom/features <madmom/features>`_).

Executable scripts
------------------

The package includes executable scripts in the `/bin <bin>`_ folder.
If you installed the package, they were copied to a common place.

All scripts can be run in different modes: in ``single`` file mode to process
a single audio file and write the output to STDOUT or the given output file.

    SuperFlux single INFILE [OUTFILE]

If multiple audio files should be processed, the scripts can also be run in
``batch`` mode to write the outputs to files with the given suffix.

    SuperFlux batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF INPUT FILES

If no output directory is given, the program writes the output files to same
location as the audio files.

The ``pickle`` mode can be used to store the used parameters to be able to
exactly reproduce experiments.

Please note that the script itself as well as the modes have help messages:

    ./bin/SuperFlux -h
    ./bin/SuperFlux -h single -h
    ./bin/SuperFlux -h batch -h
    ./bin/SuperFlux -h pickle -h

will give different help messages.

Note
----

Although we try to keep the API stable, the features are considered work in
progress and thus can change without prior notice. Do *NOT* expect these to
stay the same forever! If you need stable features, clone or fork this project,
set the parameters accordingly and/or pickle the processors.

Additional resources
====================

Mailing list
------------

The `mailing list <https://groups.google.com/d/forum/madmom-users>`_ should be
used to get in touch with the developers and other users. Please ask any
questions there before opening an issue.

Wiki
----

The wiki can be found here: https://github.com/CPJKU/madmom/wiki

FAQ
---

Frequently asked questions can be found here:
https://github.com/CPJKU/madmom/wiki/FAQ

Contribution
============

Issue tracker
-------------

If you find any bugs, `please check if it is a known issue
<https://github.com/CPJKU/madmom/issues>`_. If not, please try asking on the
mailing list first, before opening a new issue.

Fork the project
----------------

Please feel encouraged to fork this project, fix bugs, add new features,
enhance documentation or contribute to this project in any other way. Pull
requests are welcome!

References
==========

.. [1] *Universal Onset Detection with bidirectional Long Short-Term Memory
    Neural Networks*.
    Florian Eyben, Sebastian Böck, Björn Schuller and Alex Graves.
    Proceedings of the 11th International Society for Music Information
    Retrieval Conference (ISMIR), 2010.
.. [2] *Enhanced Beat Tracking with Context-Aware Neural Networks*.
    Sebastian Böck and Markus Schedl.
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.
.. [3] *Polyphonic Piano Note Transcription with Recurrent Neural Networks*.
    Sebastian Böck and Markus Schedl.
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.
.. [4] *Online Real-time Onset Detection with Recurrent Neural Networks*.
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl.
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx), 2012.
.. [5] *Evaluating the Online Capabilities of Onset Detection Methods*.
    Sebastian Böck, Florian Krebs and Markus Schedl.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2012.
.. [6] *Maximum Filter Vibrato Suppression for Onset Detection*.
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx), 2013.
.. [7] *Local Group Delay based Vibrato and Tremolo Suppression for Onset
    Detection*.
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [8] *Rhythmic Pattern Modelling for Beat and Downbeat Tracking in Musical
    Audio*.
    Florian Krebs, Sebastian Böck and Gerhard Widmer.
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [9] *Enhanced Peak Picking for Onset Detection with Recurrent Neural
    Networks*.
    Sebastian Böck, Jan Schlüter and Gerhard Widmer.
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.
.. [10] *A Multi-Model Approach to Beat Tracking Considering Heterogeneous
    Music Styles*.
    Sebastian Böck, Florian Krebs and Gerhard Widmer.
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [11] *Probabilistic Extraction of Beat Positions from a Beat Activation
    Function*.
    Filip Korzeniowski, Sebastian Böck and Gerhard Widmer.
    In Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [12] *Accurate Tempo Estimation based on Recurrent Neural Networks and
    Resonating Comb Filters*.
    Sebastian Böck, Florian Krebs and Gerhard Widmer.
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
.. [13] *An Efficient State Space Model for Joint Tempo and Meter Tracking*.
    Florian Krebs, Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
