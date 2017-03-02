Usage
=====

Executable programs
-------------------

The package includes executable programs in the ``/bin`` folder. These are
standalone reference implementations of the algorithms contained in the
package. If you just want to try/use these programs, please follow the
:ref:`instruction to install from a package <install_from_package>`.

All scripts can be run in different modes: in ``single`` file mode to process
a single audio file and write the output to STDOUT or the given output file::

  DBNBeatTracker single [-o OUTFILE] INFILE

If multiple audio files should be processed, the scripts can also be run in
``batch`` mode to write the outputs to files with the given suffix::

  DBNBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] FILES

If no output directory is given, the program writes the output files to same
location as the audio files.

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


Library usage
-------------

To use the library, :ref:`installing it from source <install_from_source>` is
the preferred way. Installation from package works as well, but you're limited
to the functionality provided and can't extend the library.

The basic usage is::

  import madmom
  import numpy as np

To learn more about how to use the library please follow the
:doc:`tutorials <tutorial>`.
