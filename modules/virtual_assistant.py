# modules/virtual_assistant.py
import time
from modules.logging import logger
from modules.whisper_recognizer import WhisperRecognizer
from modules.lm_client import LMStudioClient
from modules.hotword_detector import HotwordDetector
from modules.config import load_config

class VirtualAssistant:
    def __init__(self, config_path: str = "settings.yml"):
        self.config = load_config(config_path)
        self.recognizer = WhisperRecognizer(
            model_name=self.config["whisper"]["model"],
            sample_rate=self.config["whisper"]["sample_rate"],
            config=self.config
        )
        self.lm_client = LMStudioClient(self.config)
        # Always initialize hotword detector if enabled in config
        self.hotword_detector = HotwordDetector(config=self.config) if self.config["hotword"]["enabled"] else None
        self.word_limit = self.config["segmentation"]["max_words"]
        self._running = False

    def run(self):
        self._running = True
        try:
            while self._running:
                try:
                    if not self._wait_for_activation():
                        continue
                    # Record and process user input
                    user_text = self.recognizer.transcribe()
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
                except Exception as e:
                    logger.error("Error in main loop: %s", str(e))
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Exiting assistant. Goodbye!")
        finally:
            try:
                self.performance.report(force=True)
            except Exception as e:
                logger.error("Error in performance report: %s", str(e))
            self._running = False

    def stop(self):
        self._running = False

    def _flush_stdin(self):
        """Flush any lingering input from stdin."""
        try:
            import sys
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except Exception:
                pass

    def _wait_for_activation(self) -> bool:
        """
        Wait for activation either by detecting a keypress (push-to-talk)
        or by detecting the hotword, whichever comes first.
        """
        logger.info("Press ENTER or say '%s' to interact.", self.config["hotword"]["phrase"])
        hotword_timeout = self.config["hotword"]["timeout_sec"] if self.hotword_detector else 0
        elapsed = 0.0
        check_interval = 0.5
        while not self.hotword_detector or elapsed < hotword_timeout:
            if self._check_for_keypress():
                self._flush_stdin()
                return True
            if self.hotword_detector and self.hotword_detector.check_for_hotword(timeout=check_interval):
                return True
            time.sleep(check_interval)
            elapsed += check_interval
        return True

    def _check_for_keypress(self) -> bool:
        """Non-blocking keypress check."""
        try:
            import msvcrt  # Windows
            return msvcrt.kbhit()
        except ImportError:
            import sys
            import select  # Unix
            return sys.stdin in select.select([sys.stdin], [], [], 0)[0]
