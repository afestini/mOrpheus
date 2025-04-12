# modules/virtual_assistant.py
import time
from modules.logging import logger
from modules.whisper_recognizer import WhisperRecognizer
from modules.lm_client import LMStudioClient
from modules.audio import AudioStreamWatcher
from modules.config import load_config


class VirtualAssistant:
    def __init__(self, config_path: str = "settings.yml"):
        self.config = load_config(config_path)
        self.recognizer = WhisperRecognizer(
            model_name = self.config["whisper"]["model"],
            sample_rate = self.config["whisper"]["sample_rate"],
            config = self.config
        )
        self.audio_watcher = AudioStreamWatcher(
            sample_rate = self.config["whisper"]["sample_rate"],
            device = self.config["audio"]["input_device"]
        )
        self.lm_client = LMStudioClient(self.config)
        self.word_limit = self.config["segmentation"]["max_words"]
        self.hotword_enabled = self.config["hotword"]["enabled"]
        self.hotword_phrase = self.config["hotword"]["phrase"]
        self._running = False


    def run(self):
        self._running = True
        prompt = True
        try:
            while self._running:
                try:
                    user_text = ''
                    triggered = False
                    if prompt:
                        print(f"Press ENTER or say '{self.hotword_phrase}'")
                        prompt = False

                    if self.hotword_enabled:
                        triggered, user_text = self._check_hotword()
                    else:
                        triggered, _ = self.audio_watcher.check_for_keypress()

                    if not triggered:
                        continue

                    logger.info("Listening...")
                    audio = self.audio_watcher.record_until_silence()
                    if audio.size == 0:
                        continue

                    user_text += self.recognizer.transcribe(audio)
                    if not user_text:
                        logger.warning("No speech detected.")
                        continue

                    print("Input: ", user_text)
                    # Get and process response
                    response_text = ''
                    for fragment in self.lm_client.chat(user_text):
                        response_text += fragment
                        print(fragment, end = '', flush = True)
                        word_count = len(response_text.split())
                        if word_count > self.word_limit:
                            response_text = self.lm_client.synthesize_long_text(response_text)
                    print()
                    if response_text:
                        self.lm_client.synthesize_speech(response_text)
                    prompt = True
                except Exception as e:
                    logger.error("Error in main loop: %s", str(e))
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting assistant. Goodbye!")
        finally:
            self._running = False


    def stop(self):
        self._running = False


    def _check_hotword(self) -> tuple[bool, str]:
        audio = self.audio_watcher.listen_for_speech()
        if audio.size > 0:
            overheard = self.recognizer.transcribe(audio)
            idx = overheard.find(self.hotword_phrase)
            if idx < 0:
                return False, ''

            idx += len(self.hotword_phrase)
            while idx < len(overheard) and not overheard[idx].isalnum():
                idx += 1
            return True, overheard[idx:]
        return True, ''

