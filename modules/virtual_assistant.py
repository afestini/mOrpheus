# modules/virtual_assistant.py
from modules.logging import logger
from modules.whisper_recognizer import WhisperRecognizer
from modules.lm_client import LMStudioClient
from modules.audio import AudioStreamWatcher
from modules.config import load_config


class VirtualAssistant:
    def __init__(self, config_path: str = "settings.yml"):
        self.config = load_config(config_path)
        self.recognizer = WhisperRecognizer(config = self.config)
        self.audio_watcher = AudioStreamWatcher(config = self.config)
        self.lm_client = LMStudioClient(self.config)
        self.word_limit = self.config["tts"]["max_words"]
        self.wakeword_enabled = self.config["wakeword"]["enabled"]
        self.wakeword_phrase = self.config["wakeword"]["phrase"]
        self.wakeword_ci = self.wakeword_phrase.lower()
        self._running = False


    def run(self):
        self._running = True
        prompt = True
        self.audio_watcher.start()
        try:
            while self._running:
                try:
                    if prompt:
                        print(f"Press ENTER or say '{self.wakeword_phrase}'")
                        prompt = False

                    user_text = self.audio_watcher.processing_func()
                    if not user_text:
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
        except KeyboardInterrupt:
            logger.info("Exiting assistant. Goodbye!")
        finally:
            self.audio_watcher.stop()
            self._running = False


    def stop(self):
        self._running = False
