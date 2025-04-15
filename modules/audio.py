# modules/audio.py
import time
import queue
import numpy as np
import sounddevice as sd
from modules.vad import VAD, VADState
from modules.keypress import check_for_keypress
from modules.whisper_recognizer import WhisperRecognizer
from dvg_ringbuffer import RingBuffer
from enum import Enum


class State(Enum):
    WAIT_FOR_VAD = 0,
    LISTEN_FOR_WAKE_WORD = 1,
    RECORDING = 2


class AudioStreamWatcher:
    def __init__(self, config):
        sample_rate = config["whisper"]["sample_rate"]
        device = config["audio"]["input_device"]

        frame_size = 512 if sample_rate == 16000 else 256
        ms_per_frame = 1000 * frame_size // sample_rate

        self.wake_word_lc = config["wakeword"]["phrase"].lower()
        self.next_wake_word_check_time = 0

        self.recognizer = WhisperRecognizer()

        self.mic_stream = sd.InputStream(
                samplerate = sample_rate,
                channels = 1,
                device = device,
                blocksize = frame_size,
                callback = self.mic_callback)

        voice_threshold = config["vad"]["voice_confidence_threshold"]
        silence_threshold = config["vad"]["silence_confidence_threshold"]
        self.silence_threshold_ms = config["vad"]["silence_threshold_ms"]
        self.silence_threshold_wake_word_ms = config["vad"]["silence_threshold_wake_word_ms"]
        self.vad = VAD(sample_rate = sample_rate,
                       voice_threshold = voice_threshold,
                       silence_threshold = silence_threshold)

        self.recorded_frames = []
        self.recent_frames = np.zeros(sample_rate * 3, dtype=np.float32)
        self.recent_idx = 0

        self.state = State.WAIT_FOR_VAD
        self.frame_queue = queue.Queue(maxsize = 1000 // ms_per_frame)


    def start(self):
        self.mic_stream.start()


    def stop(self):
        self.mic_stream.stop()


    def mic_callback(self, indata, frames, timestamp, status):
        try:
            chunk = indata[:, 0].copy()
            self.frame_queue.put_nowait(chunk)
        except queue.Full:
            pass


    def update_recent_frames(self, chunk):
        n = len(chunk)
        idx = self.recent_idx % len(self.recent_frames)
        if idx + n < len(self.recent_frames):
            self.recent_frames[idx:idx + n] = chunk
        else:
            part1 = len(self.recent_frames) - idx
            self.recent_frames[idx:] = chunk[:part1]
            self.recent_frames[:n - part1] = chunk[part1:]
        self.recent_idx += n


    def processing_func(self):
        chunk = None
        try:
            chunk = self.frame_queue.get(timeout = 0.1)
        except queue.Empty:
            return ''

        silence_threshold = self.silence_threshold_ms if self.state == State.RECORDING else self.silence_threshold_wake_word_ms
        vad_state = self.vad.update_state(chunk, silence_threshold)

        if self.state == State.RECORDING:
            if vad_state == VADState.SILENCE:
                print("Transcribing...")
                audio = np.concatenate(self.recorded_frames).flatten()
                self.state = State.WAIT_FOR_VAD
                text = self.recognizer.transcribe(audio)
                idx = text.lower().find(self.wake_word_lc)
                if idx < 0:
                    return text
                idx += len(self.wake_word_lc)
                while idx < len(text) and not text[idx].isalnum():
                    idx += 1
                return text[idx:]
            else:
                self.recorded_frames.append(chunk)
            return ''

        if check_for_keypress():
            self.vad.set_active()
            self.recorded_frames.clear()
            print("Listening...")
            self.state = State.RECORDING

        if self.state == State.WAIT_FOR_VAD:
            self.update_recent_frames(chunk)
            if vad_state == VADState.ATTACK:
                self.next_wake_word_check_time = time.time() + .2
               # print("Listening for wake word")
                self.state = State.LISTEN_FOR_WAKE_WORD

        if self.state == State.LISTEN_FOR_WAKE_WORD:
            self.update_recent_frames(chunk)
            if time.time() >= self.next_wake_word_check_time:
                self.next_wake_word_check_time = time.time() + .2
                audio = np.roll(self.recent_frames, -self.recent_idx % len(self.recent_frames))
                if self.check_for_wake_word(audio):
                    print("Listening...")
                    self.recorded_frames = [audio]
                    self.state = State.RECORDING
            elif vad_state == VADState.SILENCE:
                print("Waiting for voice activation")
                self.state = State.WAIT_FOR_VAD
        return ''


    def check_for_wake_word(self, audio) -> bool:
        overheard = self.recognizer.transcribe(audio)
       # print(f"Overheard: '{overheard}'")
        idx = overheard.lower().find(self.wake_word_lc)
        return idx >= 0


class TTS:
    def __init__(self, config):
        sample_rate = config["sample_rate"]
        self.start_delay_ms = config["start_delay_ms"]
        self.word_limit = config["max_words"]
        self.audio_buffer = RingBuffer(capacity = sample_rate * 10, dtype = np.float32)
        self.zero_buffer = np.zeros((sample_rate,), dtype=np.float32)
        self.stream_paused = True
        self.last_data_time = 0
        self.buffering_size = int(sample_rate * config["buffer_duration_ms"] / 1000)
        self.start_delay_ms = 500


    def audio_callback(self, outdata, frames, _, status):
        if status:
            print(status)

        available = len(self.audio_buffer)
        time_since_last_push = (time.time_ns() - self.last_data_time) // 1000000
        done_waiting = available and (available > self.buffering_size or time_since_last_push > self.start_delay_ms)

        if self.stream_paused and not done_waiting:
            outdata[:, 0] = self.zero_buffer[:frames]
        elif available < frames:
            if time_since_last_push < 1000:
                print(f"Playback underflow {available} frames left")
            outdata[:, 0] = np.concatenate((self.audio_buffer[:available], self.zero_buffer[:frames - available]))
            self.audio_buffer.clear()
            self.stream_paused = True
        else:
            self.stream_paused = False
            outdata[:, 0] = self.audio_buffer[:frames]
            for _ in range(frames):
                self.audio_buffer.popleft()


    def push_samples(self, samples):
        size_left = self.audio_buffer.maxlen - len(self.audio_buffer)
        while len(samples) > size_left:
            time.sleep(.1)
            size_left = self.audio_buffer.maxlen - len(self.audio_buffer)

        self.audio_buffer.extend(samples)
        self.last_data_time = time.time_ns()
