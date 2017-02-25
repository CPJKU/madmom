
from pyaudio import PyAudio, paInt16, paContinue
import time
from madmom.processors import Processor
import numpy as np

SAMPLE_RATE = 44100


class PlayProcessorCallback(Processor):
    """
    This is to play audio in a new thread
    Parameters
    ----------
    get_frame : a function handle to know how to get the new frames to play
    sample_rate : int
        sample rate [Hz] of the signal
    frame_size : int
        size of frames to play in samples
    """
    def __init__(self, get_frame, sample_rate, frame_size):

        self.play = True
        self.get_frame = get_frame
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def callback(self, in_data, frame_count, time_info, flag):
        """
        callback function for the pyaudio stream : called until stream is over

        Parameters
        ----------
        in_data : None (no input data)
        frame_count : number of samples to be played by stream
        time_info : not needed
        flag : status of the stream

        Returns
        -------
        the signal to be played and the next flag (continue until the end)

        """
        if flag:
            print("Playback Error: %i" % flag)

        play_signal = self.get_frame()

        return play_signal.tostring(), paContinue

    def run(self):
        print("....................START PLAYING.............")
        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=1,
                         rate=self.sample_rate,
                         output=True,
                         frames_per_buffer=self.frame_size,
                         stream_callback=self.callback)
        while stream.is_active():
            if not self.play:
                break
            time.sleep(0.1)
        stream.close()
        pa.terminate()
        stream.close()
        pa.terminate()
        print("....................STOP PLAYING.............")

    def stop(self):
        self.play = False


class PlaybackProcessor(Processor):
    """
    Play audio in a new thread.

    Parameters
    ----------
    sample_rate : int
        sample rate [Hz] of the signal
    frame_size : int
        size of frames to play in samples

    """

    def __init__(self, sample_rate=SAMPLE_RATE, **kwargs):
        self.sample_rate = sample_rate

        self.pa = PyAudio()
        self.stream = self.pa.open(format=paInt16,
                                   channels=1,
                                   rate=self.sample_rate,
                                   output=True)

        bip_duration = int(round(0.01 * self.sample_rate))
        bip_ampl = 2 ** 15
        bip = bip_ampl * np.sin(2 * np.pi * 2000 / self.sample_rate *
                                np.arange(bip_duration))
        # the sound needs to be longer (otherwise it doesn't play)
        out = np.zeros(int(0.01*self.sample_rate))
        out[:bip_duration] = bip
        # converting in int16
        self.bip = bip.astype(np.int16).tostring()

    def process(self, data, **kwargs):
        if data:
            self.stream.write(self.bip)
        return data

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

def main():
    play = PlaybackProcessor(44100)

    for i in range(20):
        play.process(1)
        time.sleep(1)

if __name__ == '__main__':
    main()
