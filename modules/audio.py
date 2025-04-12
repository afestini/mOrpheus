# modules/audio.py
import time
import torch
import numpy as np
import sounddevice as sd
import collections
import silero_vad as silero
from dvg_ringbuffer import RingBuffer
from modules.logging import logger

# Default parameters (can be overridden via configuration)
SILENCE_THRESHOLD_MS = 1000
MIN_RECORD_TIME_MS = 2000


class AudioStreamWatcher:
    def __init__(self, sample_rate: int = 16000, device = None):
        self.vad = silero.load_silero_vad()
        self.sample_rate = sample_rate
        self.frame_size = 512 if sample_rate == 16000 else 256
        self.ms_per_frame = 1000 * self.frame_size // sample_rate
        self.stream = sd.InputStream(samplerate=sample_rate, channels=1, device=device, blocksize=self.frame_size)


    def check_for_keypress(self) -> bool:
        """Non-blocking keypress check."""
        pressed = False
        try:
            import msvcrt  # Windows
            while msvcrt.kbhit():
                pressed = True
                msvcrt.getch()
        except ImportError:
            import sys
            import select
            import termios
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                pressed = True
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
        return pressed


    def listen_for_speech(self):
        recorded_frames = collections.deque(maxlen = 2000 // self.ms_per_frame)

        self.stream.start()

        while True:
            frame, _ = self.stream.read(self.frame_size)
            recorded_frames.append(frame)

            if self.check_for_keypress():
                return np.empty(0)

            tensor = torch.from_numpy(frame.flatten()).float()
            if self.vad(tensor, self.sample_rate).item() > .7:
                break

        return self.record_until_silence(250 // self.ms_per_frame, 500, recorded_frames)


    def record_until_silence(self,
                             silence_frames = 0,
                             min_record_time = MIN_RECORD_TIME_MS,
                             existing_frames = []):
        """
        Records audio until a period of silence is detected.
        """
        try:
            recorded_frames = []
            recorded_frames.extend(existing_frames)
            if silence_frames == 0:
                silence_frames = SILENCE_THRESHOLD_MS / self.ms_per_frame

            consecutive_silence = 0
            start_time = time.time()
            if not self.stream.active:
                self.stream.start()

            while True:
                frame, _ = self.stream.read(self.frame_size)
                recorded_frames.append(frame)

                tensor = torch.from_numpy(frame.flatten()).float()
                is_speech  = self.vad(tensor, self.sample_rate).item() > .7
                if is_speech:
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1

                elapsed_ms = (time.time() - start_time) * 1000
                if consecutive_silence >= silence_frames and elapsed_ms > min_record_time:
                    break

            self.stream.stop()
            return np.concatenate(recorded_frames, axis=0)
        except Exception as e:
            logger.error("Error during audio recording: %s", str(e))
            raise


def segment_text(text, max_words=60):
    """
    Splits text into segments of at most max_words, attempting to split at sentence boundaries.
    """
    words = text.split()
    segments = []
    while len(words) > max_words:
        segment = " ".join(words[:max_words])
        cutoff = max_words
        for i, word in enumerate(segment.split()):
            if word.endswith((".", "!", "?")):
                cutoff = i + 1
        segments.append(" ".join(words[:cutoff]).strip())
        words = words[cutoff:]
    if words:
        segments.append(" ".join(words).strip())
    return segments


audio_buffer = RingBuffer(capacity=240000, dtype=np.float32)
zero_buffer = np.zeros((24000,), dtype=np.float32)
stream_paused = True
last_data_time = 0

def audio_callback(outdata, frames, _, status):
    global audio_buffer
    global stream_paused

    if status:
        print(status)

    available = len(audio_buffer)
    time_since_last_push = time.time_ns() - last_data_time
    done_waiting = available and (available > 24000 or time_since_last_push > 0.15)

    if stream_paused and not done_waiting:
        outdata[:, 0] = zero_buffer[:frames]
    elif available < frames:
        #print("streaming ", frames, " available ", available)
        outdata[:, 0] = np.concatenate((audio_buffer[:available], zero_buffer[:frames - available]))
        audio_buffer.clear()
        stream_paused = True
    else:
        stream_paused = False
        #print("streaming ", frames, " available ", available)
        outdata[:, 0] = audio_buffer[:frames]
        for i in range(frames):
            audio_buffer.popleft()


def push_samples(samples, sample_rate):
    global audio_buffer
    global time_since_last_push

    while len(audio_buffer) > sample_rate * 5:
        time.sleep(0.05)
    audio_buffer.extend(samples)
    time_since_last_push = time.time_ns()
