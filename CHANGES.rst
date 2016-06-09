Release Notes
=============

Version 0.13.2 (release date: 2016-06-09)
-----------------------------------------

This is a bugfix release.

* Fix custom filterbank in FilteredSpectrogram (#142)

Version 0.13.1 (release date: 2016-03-14)
-----------------------------------------

This is a bugfix release.

* Fix beat evaluation argument parsing (#116)

Version 0.13 (release date: 2016-03-07)
---------------------------------------

Main features:

* Python 3 support (3.3+) (#15)
* Online documentation available at http://madmom.readthedocs.org (#60)

Bug fixes:

* Fix nasty unsingend indexing bug (#88)
* MIDI note timing could get corrupted if `note_ticks_to_beats()` was called
  multiple times (#90)

API relevant changes:

* Renamed `DownBeatTracker` and all relevant classes to `PatternTracker` (#25)
* Complete refactoring of the `features.beats_hmm` module (#52)
* Unified negative index behaviour of `FramedSignal` (#72)
* Removed pickling of data classes since it was not tested thoroughly (#81)
* Reworked stacking of spectrogram differences (#82)
* Renamed `norm_bands` argument of `MultiBandSpectrogram` to `norm_filters`
  (#83)

Other changes:

* Added alignment evaluation (#12)
* Added continous integration testing (#16)
* Added `-o` option to both `single`/`batch` processing mode to not overwrite
  files accidentially in `single` mode (#18)
* Removed `block_size` parameter from `FilteredSpectrogram` (#22)
* Sample rate is always integer (#23)
* Converted all docstrings to the numpydoc format (#48)
* Batch processing continues if non-audio files are given (#53)
* Added code quality checks (#61)
* Added coverage measuring (#74)
* Added `--down`` option to evaluate only downbeats (#76)
* Removed option to normalise the observations (#95)
* Moved filterbank related argument parser to `FilterbankProcessor` (#96)

Version 0.12.1 (release date: 2016-01-22)
-----------------------------------------

Added Python 3 compatibility to setup.py (needed for the tutorials to work)

Version 0.12 (release date: 2015-10-16)
---------------------------------------

Initial public release of madmom
