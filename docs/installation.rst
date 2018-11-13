Installation
============

Please do not try to install from the .zip files provided by GitHub. Rather
install either install :ref:`from package <install_from_package>` (if you just
want to use it) or :ref:`from source <install_from_source>` (if you plan to
use it for development). Whichever variant you choose, please make sure that
all :ref:`prerequisites <install_prerequisites>` are installed.

.. _install_prerequisites:

Prerequisites
-------------

To install the ``madmom`` package, you must have either Python 2.7 or Python
3.3 or newer and the following packages installed:

- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `cython <http://www.cython.org>`_
- `mido <https://github.com/olemb/mido>`_ (for MIDI handling)
- `pytest <https://www.pytest.org/>`_ (to run the tests)
- `pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_ (to process live
  audio input)

If you need support for audio files other than ``.wav`` with a sample rate of
44.1kHz and 16 bit depth, you need ``ffmpeg`` (``avconv`` on Ubuntu Linux has
some decoding bugs, so we advise not to use it!).

Please refer to the ``requirements.txt`` file for the minimum required versions
and make sure that these modules are up to date, otherwise it can result in
unexpected errors or false computations!

.. _install_from_package:

Install from package
--------------------

The instructions given here should be used if you just want to install the
package, e.g. to run the bundled programs or use some functionality for your
own project. If you intend to change anything within the `madmom` package,
please follow the steps in :ref:`the next section <install_from_source>`.

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

.. _install_from_source:

Install from source
-------------------

If you plan to use the package as a developer, clone the Git repository::

  git clone --recursive https://github.com/CPJKU/madmom.git

Since the pre-trained model/data files are not included in this repository but
rather added as a Git submodule, you either have to clone the repo recursively.
This is equivalent to these steps::

  git clone  https://github.com/CPJKU/madmom.git
  cd madmom
  git submodule update --init --remote

Then you can simply install the package in development mode::

  python setup.py develop --user

To run the included tests::

  python setup.py pytest

.. _upgrading:

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
