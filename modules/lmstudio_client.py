import os
import time
import json
import wave
import requests
import threading
import asyncio
from .snac_decoder import tokens_decoder_sync

class LMStudioClient:
    """
    Interfaces with the LM Studio API for text generation (chat) and text-to-speech.
    """
    def __init__(self, config_lm_api, tts_sample_rate):
        self.api_url = config_lm_api["api_url"]
        self.text_endpoint = config_lm_api["chat"]["endpoint"]
        self.tts_endpoint = config_lm_api["tts"]["endpoint"]
        self.default_model = config_lm_api["chat"]["model"]
        self.tts_model = config_lm_api["tts"]["model"]
        self.system_prompt = config_lm_api["chat"]["system_prompt"]
        self.default_voice = config_lm_api["tts"]["default_voice"]
        self.max_tokens = config_lm_api["chat"]["max_tokens"]
        self.temperature = config_lm_api["chat"]["temperature"]
        self.top_p = config_lm_api["chat"]["top_p"]
        self.repetition_penalty = config_lm_api["chat"]["repetition_penalty"]
        self.tts_max_tokens = config_lm_api["tts"]["max_tokens"]
        self.tts_temperature = config_lm_api["tts"]["temperature"]
        self.headers = {"Content-Type": "application/json"}
        self.tts_sample_rate = tts_sample_rate

    def generate_text(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        payload = {
            "model": self.default_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repetition_penalty,
            "stream": False
        }
        url = self.api_url + self.text_endpoint
        print(f"Generating text for messages: {messages}")
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Text generation failed: {response.status_code} {response.text}")
        data = response.json()
        generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"Generated text: {generated_text}")
        return generated_text

    def synthesize_speech(self, text, voice=None, output_file=None, desired_tts_duration=20):
        voice = voice if voice else self.default_voice
        prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
        payload = {
            "model": self.tts_model,
            "prompt": prompt,
            "max_tokens": self.tts_max_tokens,
            "temperature": self.tts_temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repetition_penalty,
            "stream": True
        }
        url = self.api_url + self.tts_endpoint
        print(f"Generating speech for prompt: {prompt}")
        response = requests.post(url, headers=self.headers, json=payload, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"TTS request failed: {response.status_code} {response.text}")

        def token_generator():
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token_text = data.get("choices", [{}])[0].get("text", "")
                            yield token_text
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")

        # Decode tokens into audio bytes using our SNAC-based decoder.
        audio_bytes = tokens_decoder_sync(token_generator())
        if not output_file:
            output_file = f"outputs/{voice}_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.tts_sample_rate)
            wf.writeframes(audio_bytes)
        print(f"Audio saved to {output_file}")
        # Check the duration of the generated WAV file.
        with wave.open(output_file, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        if duration < desired_tts_duration:
            print(f"Warning: Generated audio is only {duration:.2f} seconds long. Consider increasing tts_max_tokens in your configuration.")
        return output_file
