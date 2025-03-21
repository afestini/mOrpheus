import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

class WhisperRecognizer:
    """
    Uses the Whisper model to record and transcribe audio from the microphone.
    """
    def __init__(self, model_name, sample_rate):
        print("🔊 Loading Whisper model...")
        self.model = whisper.load_model(model_name)
        self.sample_rate = sample_rate

    def transcribe(self, duration=5, device=None):
        print("\n🎙️ Listening...")
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, device=device)
        sd.wait()
        wav.write("input.wav", self.sample_rate, audio)
        print("📝 Transcribing...")
        result = self.model.transcribe("input.wav")
        text = result["text"].strip()
        print(f"👤 You said: {text}")
        return text
