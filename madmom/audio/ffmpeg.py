# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains audio handling via ffmpeg functionality.

"""

from __future__ import absolute_import, division, print_function

import tempfile
import subprocess
import os
import sys
import numpy as np

from .signal import Signal

# Python 2/3 string compatibility (like six does it)
try:
    string_types = basestring
except NameError:
    string_types = str


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
                 skip=None, max_len=None, cmd='ffmpeg'):
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
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    cmd : {'ffmpeg','avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

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
    # output options
    call.extend(["-f", str(fmt)])
    if max_len:
        # use "%f" to avoid scientific float notation
        call.extend(["-t", "%f" % float(max_len)])
    # output options
    if num_channels:
        call.extend(["-ac", str(int(num_channels))])
    if sample_rate:
        call.extend(["-ar", str(int(sample_rate))])
    call.append(output)
    return call


def decode_to_disk(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, max_len=None, outfile=None, tmp_dir=None,
                   tmp_suffix=None, cmd='ffmpeg'):
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
                            skip, max_len, cmd)
        subprocess.check_call(call)
    except Exception:
        if delete_on_fail:
            os.unlink(outfile)
        raise
    return outfile


def decode_to_pipe(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, max_len=None, buf_size=-1, cmd='ffmpeg'):
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
    if not isinstance(infile, (string_types, Signal)):
        raise ValueError("only file names or Signal instances are supported "
                         "as `infile`, not %s." % infile)
    # Note: closing the file-like object only stops decoding because ffmpeg
    #       reacts on that. A cleaner solution would be calling proc.terminate
    #       explicitly, but this is only available in Python 2.6+. proc.wait
    #       needs to be called in any case.
    call = _ffmpeg_call(infile, "pipe:1", fmt, sample_rate, num_channels, skip,
                        max_len, cmd)
    # redirect stdout to a pipe and buffer as requested
    if isinstance(infile, Signal):
        proc = subprocess.Popen(call, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=buf_size)
    else:
        proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=buf_size)
    return proc.stdout, proc


def decode_to_memory(infile, fmt='f32le', sample_rate=None, num_channels=1,
                     skip=None, max_len=None, cmd='ffmpeg'):
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
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    cmd : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

    Returns
    -------
    samples : str
        Binary string representation of the audio samples.

    """
    # check input file type
    if not isinstance(infile, (string_types, Signal)):
        raise ValueError("only file names or Signal instances are supported "
                         "as `infile`, not %s." % infile)
    # prepare decoding to pipe
    _, proc = decode_to_pipe(infile, fmt=fmt, sample_rate=sample_rate,
                             num_channels=num_channels, skip=skip,
                             max_len=max_len, cmd=cmd)
    # decode the input to memory
    if isinstance(infile, Signal):
        # Note: np.getbuffer was removed in Python 3, but Python 2 memoryviews
        #       do not have the cast() method
        try:
            signal, _ = proc.communicate(np.getbuffer(infile))
        except AttributeError:
            mv = memoryview(infile)
            signal, _ = proc.communicate(mv.cast('b'))
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
        output = subprocess.check_output([cmd, "-v", "quiet", "-show_streams",
                                          infile])
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
                     start=None, stop=None, dtype=None,
                     cmd_decode='ffmpeg', cmd_probe='ffprobe'):
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
        Reduce or expand the signal to `num_channels` channels; 'None' returns
        the signal with its original channels.
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
                                            skip=start, max_len=max_len,
                                            cmd=cmd_decode),
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
