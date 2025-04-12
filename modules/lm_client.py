# modules/lm_client.py
import lmstudio as lms
import sounddevice as sd
import re
import numpy as np
from time import sleep, time_ns
from dvg_ringbuffer import RingBuffer
from modules.audio import segment_text
from modules.snac_decoder import tokens_decoder

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


audio_buffer = RingBuffer(capacity=240000, dtype=np.float32)
zero_buffer = np.zeros((24000,), dtype=np.float32)
stream_paused = True
last_data_time = 0

def audio_callback(outdata, frames, time, status):
    global audio_buffer
    global stream_paused

    if status:
        print(status)

    available = len(audio_buffer)
    time_since_last_push = time_ns() - last_data_time
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

        self.chat_context = lms.Chat(self.system_prompt)

        self.stream = sd.OutputStream(samplerate = self.tts_sample_rate,
                                      channels = 1,
                                      callback = audio_callback)
        self.stream.start()


    def chat(self, user_input: str):
        self.chat_context.add_user_message(user_input)
        for fragment in self.lms_chat.respond_stream(self.chat_context, on_message = self.chat_context.append):
            yield fragment.content


    def synthesize_speech(self, text: str) -> str:
        global time_since_last_push
        # Clean the text for TTS
        cleaned_text = clean_text_for_tts(text)
        prompt = f"<|audio|>{self.default_voice}: {cleaned_text}<|eot_id|>"
        response_stream = self.lms_tts.complete_stream(prompt)
        for samples in tokens_decoder(response_stream):
            while len(audio_buffer) > self.tts_sample_rate * 5:
                sleep(0.05)
            audio_buffer.extend(samples)
            time_since_last_push = time_ns()


    def synthesize_long_text(self, text: str) -> str:
        word_limit = self.config["segmentation"]["max_words"]
        segments = segment_text(text, max_words = word_limit)
        for i, seg in enumerate(segments):
            if i + 1 == len(segments) and len(seg) < word_limit * 3:
                return seg
            self.synthesize_speech(seg)
        return ''
