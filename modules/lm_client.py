# modules/lm_client.py
import lmstudio as lms
import sounddevice as sd
import re
from typing import Optional
from modules.logging import logger
from modules.audio import segment_text, combine_audio_files
from modules.snac_decoder import tokens_decoder
from modules.config import load_config

def clean_text_for_tts(text: str) -> str:
    """
    Clean the text to be sent to the TTS engine by:
      • Removing newline characters and excessive whitespace.
      • Removing markdown symbols (e.g., asterisks).
      • Removing non-ASCII characters (e.g., emojis).
    """
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Remove markdown formatting
    text = re.sub(r'\*+', '', text)
    # Remove non-ASCII characters (e.g., emojis)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LMStudioClient:
    def __init__(self, config):
        self.config = config
        lm_config = config["lm"]

        # Chat parameters
        chat_config = lm_config["chat"]
        self.chat_model = chat_config["model"]
        self.system_prompt = chat_config["system_prompt"]
        self.chat_max_tokens = chat_config["max_tokens"]
        self.chat_temperature = chat_config["temperature"]
        self.chat_top_p = chat_config["top_p"]
        self.chat_repetition_penalty = chat_config["repetition_penalty"]
        self.max_response_time = chat_config["max_response_time"]

        # TTS parameters
        tts_config = lm_config["tts"]
        self.tts_model = tts_config["model"]
        self.default_voice = tts_config["default_voice"]
        self.tts_max_tokens = tts_config["max_tokens"]
        self.tts_temperature = tts_config["temperature"]
        self.tts_top_p = tts_config["top_p"]
        self.tts_repetition_penalty = tts_config["repetition_penalty"]
        self.speed = tts_config["speed"]
        self.max_segment_duration = tts_config["max_segment_duration"]
        self.retries = config["speech"]["max_retries"]
        self.tts_sample_rate = config["tts"]["sample_rate"]

        self.lms_chat = lms.llm(self.chat_model, config = {
            "max_tokens": self.chat_max_tokens,
            "temperature": self.chat_temperature,
            "top_p": self.chat_top_p,
            "repeat_penalty": self.chat_repetition_penalty
        })
        self.lms_tts = lms.llm(self.tts_model, config = {
            "max_tokens": self.tts_max_tokens,
            "temperature": self.tts_temperature,
            "top_p": self.tts_top_p,
            "repeat_penalty": self.tts_repetition_penalty,
            "speed": self.speed
        })

        self.stream = sd.OutputStream(samplerate = self.tts_sample_rate, channels = 1)
        self.stream.start()


    def chat(self, user_input: str):
        for fragment in self.lms_chat.respond_stream(user_input):
            yield fragment.content


    def synthesize_speech(self, text: str, voice: Optional[str] = None) -> str:
        voice = voice if voice else self.default_voice
        if not voice.isalpha() or len(voice) > 20:
            logger.warning("Invalid voice name '%s', using default", voice)
            voice = self.default_voice

        # Clean the text for TTS
        cleaned_text = clean_text_for_tts(text)
        prompt = f"<|audio|>{voice}: {cleaned_text}<|eot_id|>"
        def token_generator():
            for fragment in self.lms_tts.complete_stream(prompt):
                yield fragment.content

        for samples in tokens_decoder(token_generator()):
            self.stream.write(samples)


    def synthesize_long_text(self, text: str, voice: Optional[str] = None) -> str:
        word_limit = self.config["segmentation"]["max_words"]
        segments = segment_text(text, max_words = word_limit)
        for i, seg in enumerate(segments):
            if i + 1 == len(segments) and len(seg) < word_limit * 3:
                return seg
            self.synthesize_speech(seg, voice=voice)
        return ''
