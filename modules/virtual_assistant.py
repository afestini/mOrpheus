import wave
import time
import numpy as np
import sounddevice as sd
from .whisper_recognizer import WhisperRecognizer
from .lmstudio_client import LMStudioClient

class VirtualAssistant:
    """
    The main virtual assistant class that integrates Whisper, LM Studio API for chat and TTS,
    and decodes TTS tokens into audio using the SNAC-based decoder.
    """
    def __init__(self, config):
        self.recognizer = WhisperRecognizer(
            model_name=config["whisper"]["model_name"],
            sample_rate=config["whisper"]["sample_rate"]
        )
        self.lm_client = LMStudioClient(
            config_lm_api=config["lm_studio_api"],
            tts_sample_rate=config["tts"]["sample_rate"]
        )
        self.input_device = config.get("audio", {}).get("input_device", None)
        self.output_device = config.get("audio", {}).get("output_device", None)
        self.desired_tts_duration = config.get("desired_tts_duration", 20)

    def play_audio(self, filename):
        print("▶️ Playing audio...")
        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(audio_array, sample_rate, device=self.output_device)
            sd.wait()

    def get_wav_duration(self, filename):
        with wave.open(filename, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

    def run(self):
        print("\n🔄 Starting the virtual assistant. Press Ctrl+C to exit.\n")
        try:
            while True:
                user_text = self.recognizer.transcribe(duration=5, device=self.input_device)
                if not user_text.strip():
                    print("⚠️ No speech detected. Please try again.")
                    continue
                response_text = self.lm_client.generate_text(user_text)
                audio_file = self.lm_client.synthesize_speech(
                    response_text,
                    desired_tts_duration=self.desired_tts_duration
                )
                duration = self.get_wav_duration(audio_file)
                print(f"Audio duration: {duration:.2f} seconds.")
                self.play_audio(audio_file)
                print("Waiting extra 1 second after playback to ensure full audio is played.")
                time.sleep(duration + 1.0)
        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully. Goodbye!")
