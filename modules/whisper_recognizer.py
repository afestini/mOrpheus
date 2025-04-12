# modules/whisper_recognizer.py
import time
import torch
from faster_whisper import WhisperModel
from modules.logging import logger
from modules.config import load_config


class WhisperRecognizer:
    def __init__(self, model_name: str = "base", sample_rate: int = 16000, config=None):
        self.config = config if config is not None else load_config()
        logger.info("Loading Whisper model (%s)...", model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(model_name, device = device)
        self.sample_rate = sample_rate
        logger.info("Whisper model loaded on %s", device)


    def transcribe(self, audio) -> str:
        """Record and transcribe audio with error handling"""
        try:
            logger.info("Transcribing...")
            start_time = time.time()
            text = ''
            segments, _ = self.model.transcribe(audio.flatten(), vad_filter = True)
            for segment in segments:
                text += segment.text
            elapsed = time.time() - start_time
            logger.debug("Transcription took %.2f seconds", elapsed)
            return text
        except Exception as e:
            logger.error("Transcription failed: %s", str(e))
            return ""
