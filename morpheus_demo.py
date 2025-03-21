"""
morpheus_demo.py

Main entry point for the Morpheus Virtual Assistant.

This assistant uses:
  - Whisper for speech recognition.
  - LM Studio API for text generation (Gemma) and text-to-speech (Orpheus).
  - SNAC-based decoder to convert the TTS token stream into PCM audio.

After generating the TTS output, the WAV file's duration is calculated and, if below a desired threshold,
a warning is printed so that you know the TTS response might be getting cut off.
"""

import os
import sys
import time
import yaml
import json
import wave
import torch
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import requests
from transformers import pipeline
import asyncio
import threading
import queue

# --------------------------
# Load configuration from config.yaml (YAML format)
# --------------------------
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Optional desired minimum TTS duration (in seconds) to check for cutoff.
# If not set in the config, default to 20 seconds.
DESIRED_TTS_DURATION = config.get("desired_tts_duration", 20)

# --------------------------
# Monkey-Patch torch.load to use weights_only=True by default
# --------------------------
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    """
    Patch torch.load to default to weights_only=True for security.
    """
    kwargs.setdefault("weights_only", True)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# --------------------------
# SNAC-based Decoder Functions (from original orpheus-local)
# --------------------------
from snac import SNAC

# Load the SNAC model used for decoding LM Studio TTS tokens into PCM audio.
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using SNAC on device: {snac_device}")
snac_model = snac_model.to(snac_device)

def convert_to_audio(multiframe, count):
    """
    Convert a list of token frames into 16-bit PCM audio using the SNAC model.
    
    Args:
        multiframe (list): List of numeric token IDs.
        count (int): Current token count.
    
    Returns:
        bytes: 16-bit PCM audio bytes.
    """
    if len(multiframe) < 7:
        return
    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    for j in range(num_frames):
        i = 7 * j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])
        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    """
    Convert a custom token string into a numeric token ID.
    
    Args:
        token_string (str): The token string (e.g., "<custom_token_123>").
        index (int): Current token index.
    
    Returns:
        int or None: Numeric token ID, or None if conversion fails.
    """
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        print("No token found in the string")
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None

async def tokens_decoder(token_gen):
    """
    Asynchronously decode a stream of token strings into audio segments.
    
    Args:
        token_gen (async generator): Async generator yielding token strings.
    
    Yields:
        bytes: Audio segment bytes.
    """
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is None:
            continue
        if token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen):
    """
    Synchronously wrap an asynchronous token decoder to generate complete audio bytes.
    
    Args:
        syn_token_gen (generator): Synchronous generator yielding token strings.
    
    Returns:
        bytes: Concatenated audio bytes.
    """
    audio_queue = queue.Queue()
    async def async_token_gen():
        for token in syn_token_gen:
            yield token
    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel
    def run_async():
        asyncio.run(async_producer())
    thread = threading.Thread(target=run_async)
    thread.start()
    audio_segments = []
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        audio_segments.append(audio)
    thread.join()
    return b"".join(audio_segments)

# --------------------------
# Class: WhisperRecognizer
# --------------------------
class WhisperRecognizer:
    """
    Uses the Whisper model to record and transcribe audio from the microphone.
    """
    def __init__(self, model_name, sample_rate):
        print("🔊 Loading Whisper model...")
        self.model = whisper.load_model(model_name)
        self.sample_rate = sample_rate

    def transcribe(self, duration=5, device=None):
        """
        Records audio for a given duration, saves it, and transcribes it.
        
        Args:
            duration (int): Duration in seconds to record.
            device (int, optional): Input device ID.
        
        Returns:
            str: Transcribed text.
        """
        print("\n🎙️ Listening...")
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, device=device)
        sd.wait()
        wav.write("input.wav", self.sample_rate, audio)
        print("📝 Transcribing...")
        result = self.model.transcribe("input.wav")
        text = result["text"].strip()
        print(f"👤 You said: {text}")
        return text

# --------------------------
# Class: LMStudioClient
# --------------------------
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
        """
        Generates text using the LM Studio chat API.
        
        Args:
            user_input (str): The user's input text.
        
        Returns:
            str: The generated response text.
        """
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

    def synthesize_speech(self, text, voice=None, output_file=None):
        """
        Synthesizes speech using the LM Studio TTS API and decodes it via SNAC.
        
        Args:
            text (str): Text to synthesize.
            voice (str, optional): Voice to use.
            output_file (str, optional): Path for output WAV file.
        
        Returns:
            str: Path to the output WAV file.
        """
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
        # Decode tokens to audio bytes using our SNAC-based decoder.
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
        # Immediately check the duration of the generated WAV file.
        with wave.open(output_file, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        if duration < config.get("desired_tts_duration", 20):
            print(f"Warning: Generated audio is only {duration:.2f} seconds long. Consider increasing tts_max_tokens in your configuration.")
        return output_file

# --------------------------
# Class: VirtualAssistant
# --------------------------
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

    def play_audio(self, filename):
        """
        Plays the specified WAV file using the configured output device.
        
        Args:
            filename (str): Path to the WAV file.
        """
        print("▶️ Playing audio...")
        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(audio_array, sample_rate, device=self.output_device)
            sd.wait()

    def get_wav_duration(self, filename):
        """
        Calculates the duration of a WAV file.
        
        Args:
            filename (str): Path to the WAV file.
        
        Returns:
            float: Duration in seconds.
        """
        with wave.open(filename, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

    def run(self):
        """
        Runs the virtual assistant in a continuous loop:
          1. Record and transcribe user speech.
          2. Generate a response using LM Studio chat API.
          3. Synthesize TTS output and decode via SNAC.
          4. Calculate the full duration of the generated audio.
          5. Play the audio and wait for its full duration plus an extra buffer.
        """
        print("\n🔄 Starting the virtual assistant. Press Ctrl+C to exit.\n")
        try:
            while True:
                user_text = self.recognizer.transcribe(duration=5, device=self.input_device)
                if not user_text.strip():
                    print("⚠️ No speech detected. Please try again.")
                    continue
                response_text = self.lm_client.generate_text(user_text)
                audio_file = self.lm_client.synthesize_speech(response_text)
                # Calculate the duration of the output file immediately after creation.
                duration = self.get_wav_duration(audio_file)
                print(f"Audio duration: {duration:.2f} seconds.")
                self.play_audio(audio_file)
                print("Waiting extra 1 second after playback to ensure full audio is played.")
                time.sleep(duration + 1.0)
        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully. Goodbye!")

# --------------------------
# Main Entry Point
# --------------------------
if __name__ == "__main__":
    assistant = VirtualAssistant(config)
    assistant.run()
