#!/usr/bin/env python
# encoding: utf-8
"""
@author: Jan Schlüter <jan.schlueter@ofai.at>

"""

import tempfile
import subprocess
import os
import sys


def decode_to_disk(soundfile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, maxlen=None, outfile=None, tmpdir=None,
                   tmpsuffix=None):
    """
    Decodes the given audio file, optionally downmixes it to mono and
    writes it to another file as a sequence of samples.
    Returns the file name of the output file.

    :param soundfile: The sound file to decode
    :param fmt: The format of samples:
        'f32le' for float32, little-endian.
        's16le' for signed 16-bit int, little-endian.
    :param sample_rate: The sample rate to resample to.
    :param num_channels: The number of channels to reduce to.
    :param skip: Number of seconds to skip at beginning of file.
    :param maxlen: Maximum number of seconds to decode.
    :param outfile: The file to decode the sound file to. If not
        given, a temporary file will be created.
    :param tmpdir: The directory to create the temporary file in
        if no outfile was given.
    :param tmpsuffix: The file extension for the temporary file if
        no outfile was given. Example: ".pcm" (include the dot!)
    :returns: The output file name.

    """
    # create temp file if no outfile is given
    if outfile is None:
        # Looks stupid, but is recommended over tempfile.mktemp()
        f = tempfile.NamedTemporaryFile(delete=False, dir=tmpdir,
                                        suffix=tmpsuffix)
        f.close()
        outfile = f.name
        delete_on_fail = True
    else:
        delete_on_fail = False
    # call ffmpeg (throws exception on error)
    try:
        call = _assemble_ffmpeg_call(soundfile, outfile, fmt, sample_rate,
                                     num_channels, skip, maxlen)
        subprocess.check_call(call)
    except Exception:
        if delete_on_fail:
            os.unlink(outfile)
        raise
    return outfile


def decode_to_memory(soundfile, fmt='f32le', sample_rate=None, num_channels=1,
                     skip=None, maxlen=None):
    """
    Decodes the given audio file, downmixes it to mono and
    returns it as a binary string of a sequence of samples.

    :param soundfile: The sound file to decode
    :param fmt: The format of samples:
        'f32le' for float32, little-endian.
        's16le' for signed 16-bit int, little-endian.
    :param sample_rate: The sample rate to resample to.
    :param num_channels: The number of channels to reduce to.
    :param skip: Number of seconds to skip at beginning of file.
    :param maxlen: Maximum number of seconds to decode.
    :returns: A binary string of samples.

    """
    call = _assemble_ffmpeg_call(soundfile, "pipe:1", fmt, sample_rate,
                                 num_channels, skip, maxlen)
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


def decode_to_pipe(soundfile, fmt='f32le', sample_rate=None, num_channels=1,
                   skip=None, maxlen=None, bufsize=-1):
    """
    Decodes the given audio file, downmixes it to mono and returns a file-like
    object for reading the samples, as well as a process object. To stop
    decoding the file, call close() on the returned file-like object, then
    call wait() on the returned process object.

    :param soundfile: The sound file to decode
    :param fmt: The format of samples:
        'f32le' for float32, little-endian.
        's16le' for signed 16-bit int, little-endian.
    :param sample_rate: The sample rate to resample to.
    :param num_channels: The number of channels to reduce to.
    :param skip: Number of seconds to skip at beginning of file.
    :param maxlen: Maximum number of seconds to decode.
    :param bufsize: Size of buffer for the file-like object.
        -1 means OS default, 0 means unbuffered, 1 means line-buffered, any
        other value is the buffer size in bytes.
    :returns: A file-like object for reading the decoded samples, and a process
        object for the decoding process.

    """
    # Implementation note: Closing the file-like object only stops decoding
    # because ffmpeg reacts on that. A cleaner solution would be calling
    # proc.terminate explicitly, but this is only available in Python 2.6+.
    # proc.wait needs to be called in any case.
    call = _assemble_ffmpeg_call(soundfile, "pipe:1", fmt, sample_rate,
                                 num_channels, skip, maxlen)
    proc = subprocess.Popen(call,
            stdout=subprocess.PIPE,  # stdout is redirected to a pipe
            bufsize=bufsize,  # the pipe is buffered as requested
            )
    return proc.stdout, proc


def _assemble_ffmpeg_call(infile, output, fmt='f32le', sample_rate=None,
                          num_channels=1, skip=None, maxlen=None):
    """
    Internal function. Creates a sequence of strings indicating the application
    (ffmpeg) to be called as well as the parameters necessary to decode the
    given input file to the given format, at the given offset and for the given
    length to the given output.

    """
    if isinstance(infile, unicode):
        infile = infile.encode(sys.getfilesystemencoding())
    else:
        infile = str(infile)
    call = ["ffmpeg", "-v", "quiet", "-y", "-i", infile, "-f", str(fmt)]
    if num_channels is not None:
        call.extend(["-ac", str(num_channels)])
    if sample_rate is not None:
        call.extend(["-ar", str(sample_rate)])
    if skip is not None:
        call.extend(["-ss", str(float(skip))])
    if maxlen is not None:
        call.extend(["-t", str(float(maxlen))])
    call.append(output)
    return call


## FIXME: remove this class or make it fit into the new inheritance scheme
#class FFmpegFile(FramedAudio):
#    """
#    FFmpegFile takes an audio file, decodes it to memory and provides it in
#    samples or frames.
#
#    """
#
#    def __init__(self, filename, sample_rate=44100, num_channels=1,
#                 frame_size=2048, hop_size=441.0, online=False):
#        """
#        Creates a new FFmpegFile object instance.
#
#        :param filename:     name of the audio file to decode
#        :param sample_rate:  sample_rate to re-sample the file to
#        :param num_channels: number of channels to reduce the file to
#        :param frame_size:   size of one frame [default=2048]
#        :param hop_size:     progress N samples between adjacent frames
#                             [default=441.0]
#        :param online:       use only past information [default=False]
#
#        """
#        # init variables
#        self.filename = filename        # the name of the file
#        # decode the file to memory
#        signal = decode_to_memory(filename, sample_rate=sample_rate,
#                                  num_channels=num_channels)
#        signal = np.frombuffer(signal, dtype=np.float32)
#        if num_channels > 1:
#            raise NotImplementedError("Check which versions is correct")
#            signal = signal.reshape((num_channels, -1))
#            #signal = signal.reshape((-1, num_channels)).T
#        # instantiate a FramedAudio object
#        super(FFmpegFile, self).__init__(signal, sample_rate, frame_size,
#                                         hop_size, online)
