import torch
import silero_vad as silero
from enum import Enum


class VADState(Enum):
    ATTACK = 0,
    ACTIVE = 1,
    SILENCE = 2


class VAD:
    def __init__(self, sample_rate, silence_threshold_ms):
        frame_size = 512 if sample_rate == 16000 else 256
        ms_per_frame = 1000 * frame_size // sample_rate
        self.vad = silero.load_silero_vad()
        self.sample_rate = sample_rate
        self.silent_frames = 0
        self.silence_threshold = silence_threshold_ms / ms_per_frame
        self.state = VADState.SILENCE


    def set_active(self):
        self.silent_frames = 0
        self.state = VADState.ACTIVE


    def update_state(self, chunk):
        tensor = torch.from_numpy(chunk).float()
        vad_confidence = self.vad(tensor, self.sample_rate).item()

        if vad_confidence > .7:
            self.silent_frames = 0
            self.state = VADState.ATTACK if self.state == VADState.SILENCE else VADState.ACTIVE
        elif vad_confidence < .5:
            self.silent_frames += 1
            if self.silent_frames > self.silence_threshold:
                self.state = VADState.SILENCE

        return self.state
