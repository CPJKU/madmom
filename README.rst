======
madmom
======

Madmom is an audio signal processing library written in Python with a strong
focus on music information retrieval (MIR) tasks.

The library is internally used by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

Possible acronyms are:

- Madmom Analyzes Digitized Music Of Musicians
- Mostly Audio / Dominantly Music Oriented Modules

It includes reference implementations for some music information retrieval
algorithms, please see the `References`_ section.


Documentation
=============

Documentation of the package can be found online http://madmom.readthedocs.org


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

Please do not try to install from the .zip files provided by GitHub. Rather
install it from package (if you just want to use it) or source (if you plan to
use it for development) by following the instructions below. Whichever variant
you choose, please make sure that all prerequisites are installed.

Prerequisites
-------------

To install the ``madmom`` package, you must have either Python 2.7 or Python
3.3 or newer and the following packages installed:

- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `cython <http://www.cython.org>`_
- `nose <https://github.com/nose-devs/nose>`_ (to run the tests)
- `pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_ (to process live
  audio input)

If you need support for audio files other than ``.wav`` with a sample rate of
44.1kHz and 16 bit depth, you need ``ffmpeg`` (``avconv`` on Ubuntu Linux has
some decoding bugs, so we advise not to use it!).

Please refer to the `requirements.txt <requirements.txt>`_ file for the minimum
required versions and make sure that these modules are up to date, otherwise it
can result in unexpected errors or false computations!

Install from package
--------------------

The instructions given here should be used if you just want to install the
package, e.g. to run the bundled programs or use some functionality for your
own project. If you intend to change anything within the `madmom` package,
please follow the steps in the next section.

The easiest way to install the package is via ``pip`` from the `PyPI (Python
Package Index) <https://pypi.python.org/pypi>`_::

    pip install madmom

This includes the latest code and trained models and will install all
dependencies automatically.

You might need higher privileges (use su or sudo) to install the package, model
files and scripts globally. Alternatively you can install the package locally
(i.e. only for you) by adding the ``--user`` argument::

    pip install --user madmom

This will also install the executable programs to a common place (e.g.
``/usr/local/bin``), which should be in your ``$PATH`` already. If you
installed the package locally, the programs will be copied to a folder which
might not be included in your ``$PATH`` (e.g. ``~/Library/Python/2.7/bin``
on Mac OS X or ``~/.local/bin`` on Ubuntu Linux, ``pip`` will tell you). Thus
the programs need to be called explicitely or you can add their install path
to your ``$PATH`` environment variable::

    export PATH='path/to/scripts':$PATH

Install from source
-------------------

If you plan to use the package as a developer, clone the Git repository::

    git clone --recursive https://github.com/CPJKU/madmom.git

Since the pre-trained model/data files are not included in this repository but
rather added as a Git submodule, you either have to clone the repo recursively.
This is equivalent to these steps::

    git clone https://github.com/CPJKU/madmom.git
    cd madmom
    git submodule update --init --remote

Then you can simply install the package in development mode::

    python setup.py develop --user

To run the included tests::

    python setup.py test

Upgrade of existing installations
---------------------------------

To upgrade the package, please use the same mechanism (pip vs. source) as you
did for installation. If you want to change from package to source, please
uninstall the package first.

Upgrade a package
~~~~~~~~~~~~~~~~~

Simply upgrade the package via pip::

    pip install --upgrade madmom [--user]

If some of the provided programs or models changed (please refer to the
CHANGELOG) you should first uninstall the package and then reinstall::

    pip uninstall madmom
    pip install madmom [--user]

Upgrade from source
~~~~~~~~~~~~~~~~~~~

Simply pull the latest sources::

    git pull

To update the models contained in the submodule::

    git submodule update

If any of the ``.pyx`` or ``.pxd`` files changed, you have to recompile the
modules with Cython::

    python setup.py build_ext --inplace

Package structure
-----------------

The package has a very simple structure, divided into the following folders:

`/bin <bin>`_
  this folder includes example programs (i.e. executable algorithms)
`/docs <docs>`_
  package documentation
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
`/madmom/models <../../../madmom_models>`_
  pre-trained model/data files (see the License section)
`/madmom/utils <madmom/utils>`_
  misc stuff (e.g. MIDI and general file handling)
`/tests <tests>`_
  tests

Executable programs
-------------------

The package includes executable programs in the `/bin <bin>`_ folder.
If you installed the package, they were copied to a common place.

All scripts can be run in different modes: in ``single`` file mode to process
a single audio file and write the output to STDOUT or the given output file::

    DBNBeatTracker single [-o OUTFILE] INFILE

If multiple audio files should be processed, the scripts can also be run in
``batch`` mode to write the outputs to files with the given suffix::

    DBNBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] FILES

If no output directory is given, the program writes the output files to the
same location as the audio files.

Some programs can also be run in ``online`` mode, i.e. operate on live audio
signals. This requires `pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_
to be installed::

    DBNBeatTracker online [-o OUTFILE] [INFILE]

The ``pickle`` mode can be used to store the used parameters to be able to
exactly reproduce experiments.

Please note that the program itself as well as the modes have help messages::

    DBNBeatTracker -h

    DBNBeatTracker single -h

    DBNBeatTracker batch -h

    DBNBeatTracker online -h

    DBNBeatTracker pickle -h

will give different help messages.


Additional resources
====================

Mailing list
------------

The `mailing list <https://groups.google.com/d/forum/madmom-users>`_ should be
used to get in touch with the developers and other users.

Wiki
----

The wiki can be found here: https://github.com/CPJKU/madmom/wiki

FAQ
---

Frequently asked questions can be found here:
https://github.com/CPJKU/madmom/wiki/FAQ

Citation
========

If you use madmom in your work, please consider citing it:

.. code-block:: latex

   @inproceedings{madmom,
      Title = {{madmom: a new Python Audio and Music Signal Processing Library}},
      Author = {B{\"o}ck, Sebastian and Korzeniowski, Filip and Schl{\"u}ter, Jan and Krebs, Florian and Widmer, Gerhard},
      Booktitle = {Proceedings of the 24th ACM International Conference on
      Multimedia},
      Month = {10},
      Year = {2016},
      Pages = {1174--1178},
      Address = {Amsterdam, The Netherlands},
      Doi = {10.1145/2964284.2973795}
   }

References
==========

.. [1] Florian Eyben, Sebastian Böck, Björn Schuller and Alex Graves,
    *Universal Onset Detection with bidirectional Long Short-Term Memory
    Neural Networks*,
    Proceedings of the 11th International Society for Music Information
    Retrieval Conference (ISMIR), 2010.
.. [2] Sebastian Böck and Markus Schedl,
    *Enhanced Beat Tracking with Context-Aware Neural Networks*,
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.
.. [3] Sebastian Böck and Markus Schedl,
    *Polyphonic Piano Note Transcription with Recurrent Neural Networks*,
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.
.. [4] Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl,
    *Online Real-time Onset Detection with Recurrent Neural Networks*,
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx), 2012.
.. [5] Sebastian Böck, Florian Krebs and Markus Schedl,
    *Evaluating the Online Capabilities of Onset Detection Methods*,
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2012.
.. [6] Sebastian Böck and Gerhard Widmer,
    *Maximum Filter Vibrato Suppression for Onset Detection*,
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx), 2013.
.. [7] Sebastian Böck and Gerhard Widmer,
    *Local Group Delay based Vibrato and Tremolo Suppression for Onset
    Detection*,
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [8] Florian Krebs, Sebastian Böck and Gerhard Widmer,
    *Rhythmic Pattern Modelling for Beat and Downbeat Tracking in Musical
    Audio*,
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [9] Sebastian Böck, Jan Schlüter and Gerhard Widmer,
    *Enhanced Peak Picking for Onset Detection with Recurrent Neural Networks*,
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.
.. [10] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music
    Styles*,
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [11] Filip Korzeniowski, Sebastian Böck and Gerhard Widmer,
    *Probabilistic Extraction of Beat Positions from a Beat Activation
    Function*,
    In Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [12] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *Accurate Tempo Estimation based on Recurrent Neural Networks and
    Resonating Comb Filters*,
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
.. [13] Florian Krebs, Sebastian Böck and Gerhard Widmer,
    *An Efficient State Space Model for Joint Tempo and Meter Tracking*,
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
.. [14] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *Joint Beat and Downbeat Tracking with Recurrent Neural Networks*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [15] Filip Korzeniowski and Gerhard Widmer,
    *Feature Learning for Chord Recognition: The Deep Chroma Extractor*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [16] Florian Krebs, Sebastian Böck, Matthias Dorfer and Gerhard Widmer,
    *Downbeat Tracking Using Beat-Synchronous Features and Recurrent Networks*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [17] Filip Korzeniowski and Gerhard Widmer,
    *A Fully Convolutional Deep Auditory Model for Musical Chord Recognition*,
    Proceedings of IEEE International Workshop on Machine Learning for Signal
    Processing (MLSP), 2016.


Acknowledgements
================

Supported by the European Commission through the `GiantSteps project
<http://www.giantsteps-project.eu>`_ (FP7 grant agreement no. 610591) and the
`Phenicx project <http://phenicx.upf.edu>`_ (FP7 grant agreement no. 601166)
as well as the `Austrian Science Fund (FWF) <https://www.fwf.ac.at>`_ project
Z159.
