# modules/audio.py
import time
import numpy as np
import sounddevice as sd
import webrtcvad
from dvg_ringbuffer import RingBuffer
from modules.logging import logger

# Default parameters (can be overridden via configuration)
VAD_MODE = 2
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD_MS = 1000
MIN_RECORD_TIME_MS = 2000


def record_until_silence(sample_rate, device=None):
    global silence
    """
    Records audio until a period of silence is detected.
    """
    try:
        vad_inst = webrtcvad.Vad(VAD_MODE)
        frame_count = int(sample_rate * FRAME_DURATION_MS / 1000)
        silence_frames = int(SILENCE_THRESHOLD_MS / FRAME_DURATION_MS)
        recorded_frames = []
        consecutive_silence = 0
        start_time = time.time()
        active_frames = 0
        total_frames = 0

        with sd.InputStream(samplerate=sample_rate, channels=1, device=device, blocksize=frame_count) as stream:
            while True:
                frame, _ = stream.read(frame_count)
                frame_int16 = (np.squeeze(frame) * 32767).astype(np.int16).tobytes()
                is_speech = vad_inst.is_speech(frame_int16, sample_rate)
                total_frames += 1
                if is_speech:
                    consecutive_silence = 0
                    active_frames += 1
                    recorded_frames.append(frame)
                else:
                    consecutive_silence += 1

                elapsed_ms = (time.time() - start_time) * 1000
                if consecutive_silence >= silence_frames and elapsed_ms > MIN_RECORD_TIME_MS:
                    break

        percent_active = active_frames / total_frames
        return np.concatenate(recorded_frames, axis=0) if percent_active > 0 else np.empty(0)
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
