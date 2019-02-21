# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains audio input/output functionality.

"""

from __future__ import absolute_import, division, print_function

import errno
import os
import subprocess
import sys
import tempfile
import io

import numpy as np

from ..utils import string_types
from ..audio.signal import Signal


# error classes
class LoadAudioFileError(Exception):
    """
    Exception to be raised whenever an audio file could not be loaded.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, value=None):
        if value is None:
            value = 'Could not load audio file.'
        self.value = value

    def __str__(self):
        return repr(self.value)


# functions for loading audio files with ffmpeg
def _ffmpeg_fmt(dtype):
    """
    Convert numpy dtypes to format strings understood by ffmpeg.

    Parameters
    ----------
    dtype : numpy dtype
        Data type to be converted.

    Returns
    -------
    str
        ffmpeg format string.

    """
    # convert dtype to sample type
    dtype = np.dtype(dtype)
    # Note: list with all ffmpeg PCM sample types: ffmpeg -formats | grep PCM
    # - unsigned int, signed int, floating point:
    fmt = {'u': 'u', 'i': 's', 'f': 'f'}.get(dtype.kind)
    # - sample size in bits:
    fmt += str(8 * dtype.itemsize)
    # - little endian or big endian:
    if dtype.byteorder == '=':
        fmt += sys.byteorder[0] + 'e'
    else:
        fmt += {'|': '', '<': 'le', '>': 'be'}.get(dtype.byteorder)
    return str(fmt)


def _ffmpeg_call(infile, output, fmt='f32le', sample_rate=None, num_channels=1,
                 channel=None, skip=None, max_len=None, cmd='ffmpeg',
                 replaygain_mode=None, replaygain_preamp=0.0):
    """
    Create a sequence of strings indicating ffmpeg how to be called as well as
    the parameters necessary to decode the given input (file) to the given
    format, at the given offset and for the given length to the given output.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    output : str
        Where to decode to.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    cmd : {'ffmpeg','avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    list
        ffmpeg call.

    Notes
    -----
    'avconv' rounds decoding positions and decodes in blocks of 4096 length
    resulting in incorrect start and stop positions. Thus it should only be
    used to decode complete files.

    """
    # Note: avconv rounds decoding positions and decodes in blocks of 4096
    #       length resulting in incorrect start and stop positions
    if cmd == 'avconv' and skip is not None and max_len is not None:
        raise RuntimeError('avconv has a bug, which results in wrong audio '
                           'slices! Decode the audio files to .wav first or '
                           'use ffmpeg.')
    # input type handling
    if isinstance(infile, Signal):
        in_fmt = _ffmpeg_fmt(infile.dtype)
        in_ac = str(int(infile.num_channels))
        in_ar = str(int(infile.sample_rate))
        infile = str("pipe:0")
    elif isinstance(infile, io.IOBase):
        infile = "-"
    else:
        infile = str(infile)
    # general options
    call = [cmd, "-v", "quiet", "-y"]
    # input options
    if skip:
        # use "%f" to avoid scientific float notation
        call.extend(["-ss", "%f" % float(skip)])
    # if we decode from STDIN, the format must be specified
    if infile == "pipe:0":
        call.extend(["-f", in_fmt, "-ac", in_ac, "-ar", in_ar])
    call.extend(["-i", infile])
    if replaygain_mode:
        audio_filter = ("volume=replaygain=%s:replaygain_preamp=%.1f"
                        % (replaygain_mode, replaygain_preamp))
        call.extend(["-af", audio_filter])
    # output options
    call.extend(["-f", str(fmt)])
    if max_len:
        # use "%f" to avoid scientific float notation
        call.extend(["-t", "%f" % float(max_len)])
    # output options
    if num_channels:
        call.extend(["-ac", str(int(num_channels))])
    if channel is not None and (num_channels == 1 or num_channels is None):
        # Calling with channel=x and num_channels
        call.extend(["-af", "pan=mono|c0=c%d" % int(channel)])
    if sample_rate:
        call.extend(["-ar", str(int(sample_rate))])
    call.append(output)
    return call


def decode_to_disk(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   channel=None, skip=None, max_len=None, outfile=None,
                   tmp_dir=None, tmp_suffix=None, cmd='ffmpeg',
                   replaygain_mode=None, replaygain_preamp=0.0):
    """
    Decode the given audio file to another file.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    outfile : str, optional
        The file to decode the sound file to; if not given, a temporary file
        will be created.
    tmp_dir : str, optional
        The directory to create the temporary file in (if no `outfile` is
        given).
    tmp_suffix : str, optional
        The file suffix for the temporary file if no `outfile` is given; e.g.
        ".pcm" (including the dot).
    cmd : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    outfile : str
        The output file name.

    """
    # check input file type
    if not isinstance(infile, string_types):
        raise ValueError("only file names are supported as `infile`, not %s."
                         % infile)
    # create temp file if no outfile is given
    if outfile is None:
        # looks stupid, but is recommended over tempfile.mktemp()
        f = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir,
                                        suffix=tmp_suffix)
        f.close()
        outfile = f.name
        delete_on_fail = True
    else:
        delete_on_fail = False
    # check output file type
    if not isinstance(outfile, string_types):
        raise ValueError("only file names are supported as `outfile`, not %s."
                         % outfile)
    # call ffmpeg (throws exception on error)
    try:
        call = _ffmpeg_call(infile, outfile, fmt, sample_rate, num_channels,
                            channel, skip, max_len, cmd,
                            replaygain_mode=replaygain_mode,
                            replaygain_preamp=replaygain_preamp)
        subprocess.check_call(call)
    except Exception:
        if delete_on_fail:
            os.unlink(outfile)
        raise
    return outfile


def decode_to_pipe(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   channel=None, skip=None, max_len=None, buf_size=-1,
                   cmd='ffmpeg', replaygain_mode=None, replaygain_preamp=0.0):
    """
    Decode the given audio and return a file-like object for reading the
    samples, as well as a process object.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    buf_size : int, optional
        Size of buffer for the file-like object:
        - '-1' means OS default (default),
        - '0' means unbuffered,
        - '1' means line-buffered, any other value is the buffer size in bytes.
    cmd : {'ffmpeg','avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    pipe : file-like object
        File-like object for reading the decoded samples.
    proc : process object
        Process object for the decoding process.

    Notes
    -----
    To stop decoding the file, call close() on the returned file-like object,
    then call wait() on the returned process object.

    """
    # check input file type
    if not isinstance(infile, (string_types, io.IOBase, Signal)):
        raise ValueError("only file names, file objects or Signal instances "
                         "are supported as `infile`, not %s." % infile)
    # Note: closing the file-like object only stops decoding because ffmpeg
    #       reacts on that. A cleaner solution would be calling proc.terminate
    #       explicitly, but this is only available in Python 2.6+. proc.wait
    #       needs to be called in any case.
    call = _ffmpeg_call(infile, "pipe:1", fmt, sample_rate, num_channels,
                        channel, skip, max_len, cmd,
                        replaygain_mode=replaygain_mode,
                        replaygain_preamp=replaygain_preamp)
    # redirect stdout to a pipe and buffer as requested
    if isinstance(infile, (Signal, io.IOBase)):
        proc = subprocess.Popen(call, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=buf_size)
    else:
        proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=buf_size)
    return proc.stdout, proc


def decode_to_memory(infile, fmt='f32le', sample_rate=None, num_channels=1,
                     channel=None, skip=None, max_len=None, cmd='ffmpeg',
                     replaygain_mode=None, replaygain_preamp=0.0):
    """
    Decode the given audio and return it as a binary string representation.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    cmd : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    samples : str
        Binary string representation of the audio samples.

    """
    # check input file type
    if not isinstance(infile, (string_types, io.IOBase, Signal)):
        raise ValueError("only file names, file objects or Signal instances "
                         "are supported as `infile`, not %s." % infile)
    # prepare decoding to pipe
    _, proc = decode_to_pipe(infile, fmt=fmt, sample_rate=sample_rate,
                             num_channels=num_channels, channel=channel,
                             skip=skip, max_len=max_len, cmd=cmd,
                             replaygain_mode=replaygain_mode,
                             replaygain_preamp=replaygain_preamp)
    # decode the input to memory
    if isinstance(infile, Signal):
        # Note: np.getbuffer was removed in Python 3, but Python 2 memoryviews
        #       do not have the cast() method
        try:
            signal, _ = proc.communicate(np.getbuffer(infile))
        except AttributeError:
            mv = memoryview(infile)
            signal, _ = proc.communicate(mv.cast('b'))
    elif isinstance(infile, io.IOBase):
        signal, _ = proc.communicate(infile.read())
        infile.seek(0)
    else:
        signal, _ = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return signal


def get_file_info(infile, cmd='ffprobe'):
    """
    Extract and return information about audio files.

    Parameters
    ----------
    infile : str
        Name of the audio file.
    cmd : {'ffprobe', 'avprobe'}, optional
        Probing command (defaults to ffprobe, alternatively supports avprobe).

    Returns
    -------
    dict
        Audio file information.

    """
    # init dictionary
    info = {'num_channels': None, 'sample_rate': None}
    if isinstance(infile, Signal):
        info['num_channels'] = infile.num_channels
        info['sample_rate'] = infile.sample_rate
    else:
        # call ffprobe
        if isinstance(infile, io.IOBase):
            call = [cmd, "-v", "quiet", "-show_streams", "pipe:0"]
            proc = subprocess.Popen(call, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
            output, _ = proc.communicate(infile.read())
            retcode = proc.poll()
            infile.seek(0)
            if retcode:
                raise subprocess.CalledProcessError(retcode, call,
                                                    output=output)
        else:
            output = subprocess.check_output([cmd, "-v", "quiet",
                                              "-show_streams", infile])
        # parse information
        for line in output.split():
            if line.startswith(b'channels='):
                info['num_channels'] = int(line[len('channels='):])
            if line.startswith(b'sample_rate='):
                # the int(float(...)) conversion is necessary because
                # avprobe returns sample_rate as floating point number
                # which int() can't handle.
                info['sample_rate'] = int(float(line[len('sample_rate='):]))
    # return the dictionary
    return info


def load_ffmpeg_file(filename, sample_rate=None, num_channels=None,
                     channel=None, start=None, stop=None, dtype=None,
                     cmd_decode='ffmpeg', cmd_probe='ffprobe',
                     replaygain_mode=None, replaygain_preamp=0.0):
    """
    Load the audio data from the given file and return it as a numpy array.

    This uses ffmpeg (or avconv) and thus supports a lot of different file
    formats, resampling and channel conversions. The file will be fully decoded
    into memory if no start and stop positions are given.

    Parameters
    ----------
    filename : str
        Name of the audio sound file to load.
    sample_rate : int, optional
        Sample rate to re-sample the signal to [Hz]; 'None' returns the signal
        in its original rate.
    num_channels : int, optional
        Reduce or expand the signal to `num_channels` channels.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    dtype : numpy dtype, optional
        Numpy dtype to return the signal in (supports signed and unsigned
        8/16/32-bit integers, and single and double precision floats,
        each in little or big endian). If 'None', np.int16 is used.
    cmd_decode : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).
    cmd_probe : {'ffprobe', 'avprobe'}, optional
        Probing command (defaults to ffprobe, alternatively supports avprobe).
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    signal : numpy array
        Audio samples.
    sample_rate : int
        Sample rate of the audio samples.

    """
    # set default dtype
    if dtype is None:
        dtype = np.int16
    # ffmpeg output format
    fmt = _ffmpeg_fmt(dtype)
    # start and stop position
    if start is None:
        start = 0
    max_len = None
    if stop is not None:
        max_len = stop - start
    # convert the audio signal using ffmpeg
    signal = np.frombuffer(decode_to_memory(filename, fmt=fmt,
                                            sample_rate=sample_rate,
                                            num_channels=num_channels,
                                            channel=channel,
                                            skip=start, max_len=max_len,
                                            cmd=cmd_decode,
                                            replaygain_mode=replaygain_mode,
                                            replaygain_preamp=replaygain_preamp
                                            ),
                           dtype=dtype)
    # get the needed information from the file
    if sample_rate is None or num_channels is None:
        info = get_file_info(filename, cmd=cmd_probe)
        if sample_rate is None:
            sample_rate = info['sample_rate']
        if num_channels is None:
            num_channels = info['num_channels']
    # reshape the audio signal
    if num_channels > 1:
        signal = signal.reshape((-1, num_channels))
    return signal, sample_rate


# functions for loading/saving wave files
def load_wave_file(filename, sample_rate=None, num_channels=None, channel=None,
                   start=None, stop=None, dtype=None):
    """
    Load the audio data from the given file and return it as a numpy array.

    Only supports wave files, does not support re-sampling or arbitrary
    channel number conversions. Reads the data as a memory-mapped file with
    copy-on-write semantics to defer I/O costs until needed.

    Parameters
    ----------
    filename : str
        Name of the file.
    sample_rate : int, optional
        Desired sample rate of the signal [Hz], or 'None' to return the
        signal in its original rate.
    num_channels : int, optional
        Reduce or expand the signal to `num_channels` channels
        If 'None', return the signal with its original channels,
        or whichever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].


    Returns
    -------
    signal : numpy array
        Audio signal.
    sample_rate : int
        Sample rate of the signal [Hz].

    Notes
    -----
    The `start` and `stop` positions are rounded to the closest sample; the
    sample corresponding to the `stop` value is not returned, thus consecutive
    segment starting with the previous `stop` can be concatenated to obtain
    the original signal without gaps or overlaps.

    """
    from scipy.io import wavfile
    file_sample_rate, signal = wavfile.read(filename, mmap=True)
    # if the sample rate is not the desired one, raise exception
    if sample_rate is not None and sample_rate != file_sample_rate:
        raise ValueError('Requested sample rate of %f Hz, but got %f Hz and '
                         're-sampling is not implemented.' %
                         (sample_rate, file_sample_rate))
    # same for the data type
    if dtype is not None and signal.dtype != dtype:
        raise ValueError('Requested dtype %s, but got %s and re-scaling is '
                         'not implemented.' % (dtype, signal.dtype))
    # only request the desired part of the signal
    if start is not None:
        start = int(start * file_sample_rate)
    if stop is not None:
        stop = min(len(signal), int(stop * file_sample_rate))
    if start is not None or stop is not None:
        signal = signal[start: stop]
    if channel is not None and num_channels is None:
        # It's clear what the caller means here
        num_channels = 1
    if num_channels is not None:
        from ..audio.signal import remix
        signal = remix(signal, num_channels, channel)
    # return the signal
    return signal, file_sample_rate


def write_wave_file(signal, filename, sample_rate=None):
    """
    Write the signal to disk as a .wav file.

    Parameters
    ----------
    signal : numpy array or Signal
        The signal to be written to file.
    filename : str
        Name of the file.
    sample_rate : int, optional
        Sample rate of the signal [Hz].

    Returns
    -------
    filename : str
        Name of the file.

    Notes
    -----
    `sample_rate` can be 'None' if `signal` is a :class:`Signal` instance. If
    set, the given `sample_rate` is used instead of the signal's sample rate.
    Must be given if `signal` is a ndarray.

    """
    from scipy.io import wavfile
    if isinstance(signal, Signal) and sample_rate is None:
        sample_rate = int(signal.sample_rate)
    wavfile.write(filename, rate=sample_rate, data=signal)
    return filename


# function for automatically determining how to open audio files
def load_audio_file(filename, sample_rate=None, num_channels=None,
                    channel=None, start=None, stop=None, dtype=None,
                    replaygain_mode=None, replaygain_preamp=0.0):
    """
    Load the audio data from the given file and return it as a numpy array.
    This tries load_wave_file() load_ffmpeg_file() (for ffmpeg and avconv).

    Parameters
    ----------
    filename : str or file handle
        Name of the file or file handle.
    sample_rate : int, optional
        Desired sample rate of the signal [Hz], or 'None' to return the
        signal in its original rate.
    num_channels : int, optional
        Reduce or expand the signal to `num_channels` channels.
        If 'None', return the signal with its original channels,
        or whatever is selected by `channel`.
    channel : int, optional
        When reducing a signal to `num_channels` of 1, use this channel,
        or 'None' to return the average across all channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].
    replaygain_mode : {None, 'track','album'}, optional
        Specify the ReplayGain volume-levelling mode (None to disable).
    replaygain_preamp : float, optional
        ReplayGain preamp volume change level (in dB).

    Returns
    -------
    signal : numpy array
        Audio signal.
    sample_rate : int
        Sample rate of the signal [Hz].

    Notes
    -----
    For wave files, the `start` and `stop` positions are rounded to the closest
    sample; the sample corresponding to the `stop` value is not returned, thus
    consecutive segment starting with the previous `stop` can be concatenated
    to obtain the original signal without gaps or overlaps.
    For all other audio files, this can not be guaranteed.

    """
    # try reading as a wave file
    error = "All attempts to load audio file %r failed." % filename
    try:
        return load_wave_file(filename, sample_rate=sample_rate,
                              num_channels=num_channels, channel=channel,
                              start=start, stop=stop, dtype=dtype)
    except ValueError:
        pass
    # not a wave file (or other sample rate requested), try ffmpeg
    try:
        return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                num_channels=num_channels, channel=channel,
                                start=start, stop=stop, dtype=dtype,
                                replaygain_mode=replaygain_mode,
                                replaygain_preamp=replaygain_preamp)
    except OSError as e:
        # if it's not a file not found error, raise it!
        if e.errno != errno.ENOENT:
            raise

        # ffmpeg is not present, try avconv
        try:
            return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                    num_channels=num_channels, channel=channel,
                                    start=start, stop=stop, dtype=dtype,
                                    cmd_decode='avconv', cmd_probe='avprobe',
                                    replaygain_mode=replaygain_mode,
                                    replaygain_preamp=replaygain_preamp)
        except OSError as e:
            if e.errno == errno.ENOENT:
                error += " Try installing ffmpeg (or avconv on Ubuntu Linux)."
            else:
                raise
        except subprocess.CalledProcessError:
            pass
    except subprocess.CalledProcessError:
        pass
    raise LoadAudioFileError(error)
