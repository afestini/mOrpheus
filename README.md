# mOrpheus Virtual Assistant Demo

This project implements the mOrpheus Virtual Assistant, which integrates speech recognition, text generation, and text-to-speech (TTS) synthesis. The assistant leverages several state-of-the-art components:

- **Whisper** for speech recognition.
- **LM Studio API** for both text generation (chat) and text-to-speech synthesis.
- **SNAC-based decoder** to convert TTS token streams into PCM audio.

## Features

- **Speech Recognition:**  
  Captures audio from the microphone and transcribes it using the Whisper model.

- **Text Generation:**  
  Uses LM Studio’s chat API to generate natural language responses based on transcribed input.

- **Text-to-Speech:**  
  Converts the generated text into speech via LM Studio’s TTS API. The text is cleaned (removing newlines, markdown symbols, and emojis) before being sent to the TTS engine, and the resulting token stream is decoded using a SNAC-based decoder.
  

- **Audio Playback & Activation:**  
  Plays the generated audio and waits for the next activation. Long responses are segmentented then recombined at playback. The assistant supports both push‑to‑talk (keypress) and hotword (“Hey Assistant”) activation, or both simultaneously, as configured.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Whisper](https://github.com/openai/whisper)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [scipy](https://www.scipy.org/)
- [numpy](https://numpy.org/)
- [requests](https://docs.python-requests.org/)
- [PyYAML](https://pyyaml.org/)
- [Transformers](https://huggingface.co/transformers/)
- [SNAC](https://github.com/hubertsiuzdak/snac)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/morpheus-virtual-assistant.git
   cd morpheus-virtual-assistant
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that your `requirements.txt` includes all required packages.*

3. **Configure the application:**

   - Create a `settings.yml` file in the project root.
   - Populate it with your configuration details for Whisper, LM Studio API, audio settings, interaction mode, etc.

   Example `settings.yml`:

   ```yaml
   # -------------------------------
   # Configuration for Whisper STT
   # -------------------------------
   whisper:
     model: "small.en"
     sample_rate: 16000

   # -------------------------------
   # Configuration for LM Studio (Chat & TTS)
   # -------------------------------
   lm:
     api_url: "http://127.0.0.1:1234/v1"
     
     chat:
       endpoint: "/chat/completions"
       model: "gemma-3-12b-it"
       system_prompt: "You are a helpful assistant."
       max_tokens: 256
       temperature: 0.7
       top_p: 0.9
       repetition_penalty: 1.1
       max_response_time: 10.0

     tts:
       endpoint: "/completions"
       model: "orpheus-3b-ft.gguf@q2_k"
       default_voice: "tara"
       max_tokens: 4096
       temperature: 0.6
       top_p: 0.9
       repetition_penalty: 1.0
       speed: 1.0
       max_segment_duration: 20

   # -------------------------------
   # TTS Audio Output Configuration
   # -------------------------------
   tts:
     sample_rate: 24000

   # -------------------------------
   # Audio Device Configuration
   # -------------------------------
   audio:
     input_device: null
     output_device: null
     hotword_sample_rate: 16000

   # -------------------------------
   # Voice Activity Detection (VAD) Configuration
   # -------------------------------
   vad:
     mode: 2
     frame_duration_ms: 30
     silence_threshold_ms: 1000
     min_record_time_ms: 2000

   # -------------------------------
   # Hotword Detection Configuration
   # -------------------------------
   hotword:
     enabled: true
     phrase: "Hey Assistant"
     sensitivity: 0.7
     timeout_sec: 5
     retries: 3

   # -------------------------------
   # Segmentation Configuration for TTS
   # -------------------------------
   segmentation:
     max_words: 60

   # -------------------------------
   # Speech Quality Configuration
   # -------------------------------
   speech:
     normalize_audio: false
     default_pitch: 0
     min_speech_confidence: 0.5
     max_retries: 3

   # -------------------------------
   # Interaction Configuration
   # -------------------------------
   interaction:
     mode: "both"  # Options: "push_to_talk", "hotword", or "both"
     post_audio_delay: 0.5
   ```

4. **Run LM Studio**
Before activating the assistant you need to have LM Studio running both the LLM and Orpheus model as defined in the settings.yml in API mode. This is only accessibly in Power User or Developer Mode respectively.

5. **Run the Assistant:**

   ```bash
   python morpheus.py
   ```


## Suggested Models

These are the models I tested with, your milage may vary with additional models.

**LLM Models:**
- gemma-3-12b-it-GGUF/gemma-3-12b-it-Q3_K_L.gguf

**Orpheus Models:**
- lex-au/Orpheus-3b-FT-Q2_K.gguf - Fastest inference (~50% faster tokens/sec than Q8_0).
- lex-au/Orpheus-3b-FT-Q4_K_M.gguf - Balanced quality/speed.
- lex-au/Orpheus-3b-FT-Q8_0.gguf - Original high-quality model.

## Usage

- **Activation:**  
  The assistant listens for activation either via a hotword ("Hey Assistant") or a push-to-talk keypress (or both, depending on your settings).  
- **Speech Processing:**  
  It records your speech, transcribes it using Whisper, and generates a text response via LM Studio’s chat API.
- **TTS Synthesis:**  
  The response is cleaned to remove unwanted characters (e.g., emojis, newlines, markdown formatting) and then sent to LM Studio’s TTS API. The SNAC-based decoder converts the TTS token stream into PCM audio.
- **Audio Playback:**  
  The generated audio is played back, and a brief delay is applied before the assistant waits for the next activation.

## Contributing

Feel free to open issues or submit pull requests with improvements or bug fixes. I'm fairly new at this and feel this could see a lot of improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Notice

See the [NOTICE](NOTICE) file for details.
