# modules/lm_client.py
import lmstudio as lms
import sounddevice as sd
import re
import threading
from queue import Queue
from modules.snac_decoder import tokens_decoder
from modules.audio import TTS


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
    # Remove special tags
    text = re.sub(r'<[/]*i_[\d]+>', '', text)
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
        self.chat_use_context = chat_config["use_context"]

        # TTS parameters
        tts_config = lm_config["tts"]
        self.tts_model = tts_config["model"]
        self.default_voice = tts_config["default_voice"]
        self.tts_max_tokens = tts_config["max_tokens"]
        self.tts_temperature = tts_config["temperature"]
        self.tts_top_p = tts_config["top_p"]
        self.tts_repetition_penalty = tts_config["repetition_penalty"]
        self.speed = tts_config["speed"]
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

        self.tts = TTS(config["tts"])

        if self.chat_use_context:
            self.chat_context = lms.Chat(self.system_prompt)

        self.tts_stream = sd.OutputStream(samplerate = self.tts_sample_rate,
                                          channels = 1,
                                          callback = self.tts.audio_callback)
        self.tts_stream.start()

        self.tts_thread = threading.Thread(target = self.tts_thread_func)
        self.tts_thread.start()
        self.tts_queue = Queue(100)


    def chat(self, user_input: str):
        if self.chat_use_context:
            self.chat_context.add_user_message(user_input)
            for fragment in self.lms_chat.respond_stream(self.chat_context, on_message = self.chat_context.append):
                yield fragment.content
        else:
            for fragment in self.lms_chat.respond_stream(user_input):
                yield fragment.content


    def tts_thread_func(self):
        while self.tts_queue:
            text = None
            try:
                text = self.tts_queue.get()
            except Exception:
                continue

            prompt = f"<|audio|>{self.default_voice}: {text}<|eot_id|>"
            response_stream = self.lms_tts.complete_stream(prompt)
            for samples in tokens_decoder(response_stream):
                self.tts.push_samples(samples)


    def synthesize_speech(self, text: str) -> str:
        global time_since_last_push
        self.tts_queue.put(item = clean_text_for_tts(text))


    def synthesize_long_text(self, text: str) -> str:
        word_limit = self.config["tts"]["max_words"]
        segments = segment_text(text, max_words = word_limit)
        for i, seg in enumerate(segments):
            if i + 1 == len(segments) and len(seg) < word_limit * 3:
                return seg
            self.synthesize_speech(seg)
        return ''


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
            if word.endswith(("\n")):
                cutoff = i + 1
                break # Always split on paragraphs
            elif word.endswith((".", "!", "?")):
                cutoff = i + 1
        segments.append(" ".join(words[:cutoff]).strip())
        words = words[cutoff:]
    if words:
        segments.append(" ".join(words).strip())
    return segments
