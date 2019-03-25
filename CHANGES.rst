Release Notes
=============

Version 0.17.dev0
-----------------

New features:

* Volume changes according to `ReplayGain` tags can be applied (#400)

Bug fixes:

* `BufferProcessor` can handle data longer than buffer length (#398)

Version 0.16.1 (release date: 2017-11-14)
-----------------------------------------

This is a maintenance release.

* Include .pyx files in source distribution

Version 0.16 (release date: 2017-11-13)
---------------------------------------

New features:

* `TempoDetector` can operate on live audio signals  (#292)
* Added chord evaluation (#309)
* Bar tracking functionality (#316)
* Added `quantize_notes` function (#327)
* Added global key evaluation (#336)
* Added key recognition feature and program (#345, #381)

Bug fixes:

* Fix `TransitionModel` number of states when last state is unreachable (#287)
* Fix double beat detections in `BeatTrackingProcessor` (#298)
* Fix ffmpeg unicode filename handling (#305)
* Fix STFT zero padding (#319)
* Fix memory leak when accessing signal frames (#322)
* Quantization of events does not alter them (#327)

API relevant changes:

* `BufferProcessor` uses `data` instead of `buffer` for data storage (#292)
* `DBNBeatTrackingProcessor` expects 1D inputs (#299)
* Moved downbeat and pattern tracking to `features.downbeats` (#316)
* Write/load functions moved to `io` module (#346)
* Write functions do not return any data (#346)
* Evaluation classes expect annotations/detections, cannot handle files (#346)
* New MIDI module (io.midi) replacing (utils.midi) based on mido (#46)

Other changes:

* Viterbi decoding of `HMM` raises a warning if no valid path is found (#279)
* Add option to include Nyquist frequency in `STFT` (#280)
* Use `pyfftw` to compute FFT (#363)
* Python 3.7 support (#374)
* Use pytest instead of nose to run tests (#385)
* Removed obsolete code (#385)


Version 0.15.1 (release date: 2017-07-07)
-----------------------------------------

This is a maintenance release.

* NumPy boolean subtract fix (#296)


Version 0.15 (release date: 2017-04-25)
---------------------------------------

New features:

* Streaming mode allows framewise processing of live audio input (#185)
* Exponential linear unit (ELU) activation function (#232)
* `DBNBeatTracker` can operate on live audio signals (#238)
* `OnsetDetectorLL` can operate on live audio signals (#256)

Bug fixes:

* Fix downbeat evaluation failure with a single annotation / detection (#216)
* Fix tempo handling of multi-track MIDI files (#219)
* Fix error loading unicode filenames (#223)
* Fix ffmpeg unicode filename handling (#236)
* Fix smoothing for `peak_picking` (#247)
* Fix combining onsets/notes (#255)

API relevant changes:

* `NeuralNetwork` expect 2D inputs; activation can be computed stepwise (#244)
* Reorder `GRUCell` parameters, to be consistent with all other layers (#243)
* Rename `GRULayer` parameters, to be consistent with all other layers (#243)

Other changes:

* SPL and RMS can be computed on `Signal` and `FramedSignal` (#208)
* `num_threads` is passed to `ParallelProcessor` in single mode (#217)
* Use `install_requires` in `setup.py` to specify dependencies (#226)
* Use new Cython build system to build extensions (#227)
* Allow initialisation of previous/hidden states in RNNs (#243)
* Forward path of `HMM` can be computed stepwise (#244)


Version 0.14.1 (release date: 2016-08-01)
-----------------------------------------

This is a maintenance release.

* `RNNDownBeatProcessor` returns only beat and downbeat activations (#197)
* Update programs to reflect MIREX 2016 submissions (#198)

Version 0.14 (release date: 2016-07-28)
---------------------------------------

New features:

* Downbeat tracking based on Recurrent Neural Network (RNN) and Dynamic
  Bayesian Network (DBN) (#130)
* Convolutional Neural Networks (CNN) and CNN onset detection (#133)
* Linear-Chain Conditional Random Field (CRF) implementation (#144)
* Deep Neural Network (DNN) based chroma vector extraction (#148)
* CRF chord recognition using DNN chroma vectors (#148)
* CNN chord recognition using CRF decoding (#152)
* Initial Windows support (Python 2.7 only, no pip packages yet) (#157)
* Gated Recurrent Unit (GRU) network layer (#167)

Bug fixes:

* Fix downbeat output bug (#128)
* MIDI file creation bug (#166)

API relevant changes:

* Refactored the `ml.rnn` to `ml.nn` and converted the models to pickles (#110)
* Reordered the dimensions of comb_filters to time, freq, tau (#135)
* `write_notes` uses `delimiter` instead of `sep` to separate columns (#155)
* `LSTMLayer` takes `Gate` as arguments, all layers are callable (#161)
* Replaced `online` parameter of `FramedSignalProcessor` by `origin` (#169)

Other changes:

* Added classes for onset/note/beat detection with RNNs to `features.*` (#118)
* Add examples to docstrings of classes (#119)
* Converted `madmom.modules` into a Python package (#125)
* `match_files` can handle inexact matches (#137)
* Updated beat tracking models to MIREX 2015 ones (#146)
* Tempo and time signature can be set for created MIDI files (#166)


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

New features:

* Python 3 support (3.3+) (#15)
* Online documentation available at http://madmom.readthedocs.org (#60)

Bug fixes:

* Fix nasty unsigned indexing bug (#88)
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
* Added continuous integration testing (#16)
* Added `-o` option to both `single`/`batch` processing mode to not overwrite
  files accidentally in `single` mode (#18)
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
