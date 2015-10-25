# encoding: utf-8
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This file contains audio handling via ffmpeg functionality.

"""

from __future__ import absolute_import, division, print_function

import tempfile
import subprocess
import os
import sys
import numpy as np


def decode_to_disk(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, max_len=None, outfile=None, tmp_dir=None,
                   tmp_suffix=None, cmd='ffmpeg'):
    """
    Decodes the given audio file, optionally down-mixes it to mono and writes
    it to another file as a sequence of samples. Returns the file name of the
    output file.

    :param infile:       the audio sound file to decode
    :param fmt:          the format of the samples:
                         'f32le' for float32, little-endian
                         's16le' for signed 16-bit int, little-endian
    :param sample_rate:  the sample rate to re-sample the signal to
    :param num_channels: the number of channels to reduce the signal to
    :param skip:         number of seconds to skip at beginning of file
    :param max_len:      maximum number of seconds to decode
    :param outfile:      the file to decode the sound file to; if not given,
                         a temporary file will be created
    :param tmp_dir:      the directory to create the temporary file in
                         (if no `outfile` was given)
    :param tmp_suffix:   fhe file suffix for the temporary file if no outfile
                         was given; example: ".pcm" (including the dot!)
    :param cmd:          command line tool to use (defaults to ffmpeg,
                         alternatively supports avconv)
    :return:             the output file name

    """
    # check input and output file type
    if isinstance(infile, file):
        raise ValueError("only file names are supported as 'infile', not %s."
                         % infile)
    if isinstance(outfile, file):
        raise ValueError("only file names are supported as 'outfile', not %s."
                         % outfile)
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
    # call ffmpeg (throws exception on error)
    try:
        call = _assemble_ffmpeg_call(infile, outfile, fmt, sample_rate,
                                     num_channels, skip, max_len, cmd)
        subprocess.check_call(call)
    except Exception:
        if delete_on_fail:
            os.unlink(outfile)
        raise
    return outfile


def decode_to_memory(infile, fmt='f32le', sample_rate=None, num_channels=1,
                     skip=None, max_len=None, cmd='ffmpeg'):
    """
    Decodes the given audio file, down-mixes it to mono and returns it as a
    binary string of a sequence of samples.

    :param infile:       the audio sound file to decode
    :param fmt:          the format of the samples:
                         'f32le' for float32, little-endian
                         's16le' for signed 16-bit int, little-endian
    :param sample_rate:  the sample rate to re-sample the signal to
    :param num_channels: the number of channels to reduce the signal to
    :param skip:         number of seconds to skip at beginning of file
    :param max_len:      maximum number of seconds to decode
    :param cmd:          command line tool to use (defaults to ffmpeg,
                         alternatively supports avconv)
    :return:             a binary string of samples

    """
    # check input file type
    if not isinstance(infile, str):
        raise ValueError("only file names are supported as 'infile', not %s."
                         % infile)
    # assemble ffmpeg call
    call = _assemble_ffmpeg_call(infile, "pipe:1", fmt, sample_rate,
                                 num_channels, skip, max_len, cmd)
    if hasattr(subprocess, 'check_output'):
        # call ffmpeg (throws exception on error)
        signal = subprocess.check_output(call)
    else:
        # this is an old version of Python, do subprocess.check_output manually
        proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=-1)
        signal, _ = proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, call)
    return signal


def decode_to_pipe(infile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, max_len=None, buf_size=-1, cmd='ffmpeg'):
    """
    Decodes the given audio file, down-mixes it to mono and returns a file-like
    object for reading the samples, as well as a process object. To stop
    decoding the file, call close() on the returned file-like object, then
    call wait() on the returned process object.

    :param infile:       the audio sound file to decode
    :param fmt:          the format of the samples:
                         'f32le' for float32, little-endian
                         's16le' for signed 16-bit int, little-endian
    :param sample_rate:  the sample rate to re-sample the signal to
    :param num_channels: the number of channels to reduce the signal to
    :param skip:         number of seconds to skip at beginning of file
    :param max_len:      maximum number of seconds to decode
    :param buf_size:     size of buffer for the file-like object:
                         '-1' means OS default,
                         '0' means unbuffered,
                         '1' means line-buffered,
                         any other value is the buffer size in bytes.
    :param cmd:          command line tool to use (defaults to ffmpeg,
                         alternatively supports avconv)
    :return:             tuple (file-like object for reading the decoded
                         samples, process object for the decoding process)

    """
    # check input file type
    if isinstance(infile, file):
        raise ValueError("only file names are supported as 'infile', not %s."
                         % infile)
    # Note: closing the file-like object only stops decoding because ffmpeg
    #       reacts on that. A cleaner solution would be calling
    #       proc.terminate explicitly, but this is only available in
    #       Python 2.6+. proc.wait needs to be called in any case.
    call = _assemble_ffmpeg_call(infile, "pipe:1", fmt, sample_rate,
                                 num_channels, skip, max_len, cmd)
    # redirect stdout to a pipe and buffer as requested
    proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=buf_size)
    return proc.stdout, proc


def _assemble_ffmpeg_call(infile, output, fmt='f32le', sample_rate=None,
                          num_channels=1, skip=None, max_len=None,
                          cmd='ffmpeg'):
    """
    Internal function. Creates a sequence of strings indicating the application
    (ffmpeg) to be called as well as the parameters necessary to decode the
    given input file to the given format, at the given offset and for the given
    length to the given output.

    :param infile:       the audio sound file to decode
    :param output:       where to decode to
    :param fmt:          the format of the samples:
                         'f32le' for float32, little-endian
                         's16le' for signed 16-bit int, little-endian
    :param sample_rate:  the sample rate to re-sample to
    :param num_channels: the number of channels to reduce to
    :param skip:         number of seconds to skip at beginning of file
    :param max_len:      maximum number of seconds to decode
    :param cmd:          command line tool to use (defaults to ffmpeg,
                         alternatively supports avconv)
    :return:             assembled ffmpeg call

    """
    # Note: avconv rounds decoding positions and decodes in blocks of 4096
    #       length resulting in incorrect start and stop positions
    if cmd == 'avconv' and skip is not None and max_len is not None:
        raise RuntimeError('avconv has a bug, which results in wrong audio '
                           'slices! Decode the audio files to .wav first or '
                           'use ffmpeg.')
    if isinstance(infile, str):
        infile = infile.encode(sys.getfilesystemencoding())
    else:
        infile = str(infile)
    # general options
    call = [cmd, "-v", "quiet"]
    # infile options
    if skip is not None:
        call.extend(["-ss", "%f" % float(skip)])
    call.extend(["-i", infile, "-y", "-f", str(fmt)])
    if max_len is not None:
        call.extend(["-t", "%f" % float(max_len)])
    # output options
    if num_channels is not None:
        call.extend(["-ac", str(num_channels)])
    if sample_rate is not None:
        call.extend(["-ar", str(sample_rate)])
    call.append(output)
    return call


def get_file_info(infile, cmd='ffprobe'):
    """
    Extract and return information about audio files.

    :param infile: name of the audio file
    :param cmd:    command line tool to use (defaults to ffprobe,
                   alternatively supports avprobe)
    :return:       dictionary containing audio file information

    """
    # check input file type
    if not isinstance(infile, str):
        raise ValueError("only file names are supported as 'infile', not %s."
                         % infile)
    # init dictionary
    info = {'num_channels': None, 'sample_rate': None}
    # call ffprobe
    output = subprocess.check_output([cmd, "-v", "quiet", "-show_streams",
                                      infile])
    # parse information
    for line in output.split():
        if line.startswith(b'channels='):
            info['num_channels'] = int(line[len('channels='):])
        if line.startswith(b'sample_rate='):
            info['sample_rate'] = float(line[len('sample_rate='):])
    # return the dictionary
    return info


def load_ffmpeg_file(filename, sample_rate=None, num_channels=None,
                     start=None, stop=None, dtype=np.int16,
                     cmd_decode='ffmpeg', cmd_probe='ffprobe'):
    """
    Load the audio data from the given file and return it as a numpy array.
    This uses ffmpeg (or avconv) and thus supports a lot of different file
    formats, resampling and channel conversions. The file will be fully decoded
    into memory if no start and stop positions are given.

    :param filename:     name of the file
    :param sample_rate:  desired sample rate of the signal in Hz [int], or
                         `None` to return the signal in its original rate
    :param num_channels: reduce or expand the signal to N channels [int], or
                         `None` to return the signal with its original channels
    :param start:        start position (seconds) [float]
    :param stop:         stop position (seconds) [float]
    :param dtype:        numpy dtype to return the signal in (supports signed
                         and unsigned 8/16/32-bit integers, and single and
                         double precision floats, each in little or big endian)
    :param cmd_decode:   command used for decoding audio
    :param cmd_probe:    command used for probing audio (i.e. get file info)
    :return:             tuple (signal, sample_rate)

    """
    # convert dtype to sample type
    # (all ffmpeg PCM sample types: ffmpeg -formats | grep PCM)
    dtype = np.dtype(dtype)
    # - unsigned int, signed int, floating point:
    sample_type = {'u': 'u', 'i': 's', 'f': 'f'}.get(dtype.kind)
    # - sample size in bits:
    sample_type += str(8 * dtype.itemsize)
    # - little endian or big endian:
    if dtype.byteorder == '=':
        import sys
        sample_type += sys.byteorder[0] + 'e'
    else:
        sample_type += {'|': '', '<': 'le', '>': 'be'}.get(dtype.byteorder)
    # start and stop position
    if start is None:
        start = 0
    max_len = None
    if stop is not None:
        max_len = stop - start
    # convert the audio signal using ffmpeg
    signal = np.frombuffer(decode_to_memory(filename, fmt=sample_type,
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
