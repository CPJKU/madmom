#!/usr/bin/env python
# encoding: utf-8
"""
@author: Jan Schlüter <jan.schlueter@ofai.at>

"""

import tempfile
import subprocess
import os
import sys


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
    if isinstance(infile, file):
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
    if isinstance(infile, unicode):
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
    if isinstance(infile, file):
        raise ValueError("only file names are supported as 'infile', not %s."
                         % infile)
    # init dictionary
    info = {'num_channels': None, 'sample_rate': None}
    # call ffprobe
    output = subprocess.check_output([cmd, "-v", "quiet", "-show_streams",
                                      infile])
    # parse information
    for line in output.split():
        if line.startswith('channels='):
            info['num_channels'] = int(line[len('channels='):])
        if line.startswith('sample_rate='):
            info['sample_rate'] = float(line[len('sample_rate='):])
    # return the dictionary
    return info
