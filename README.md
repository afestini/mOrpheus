# mOrpheus Virtual Assistant Demo

This project implements the Morpheus Virtual Assistant, which integrates speech recognition, text generation, and text-to-speech (TTS) synthesis. The assistant utilizes the following components:

- **Whisper** for speech recognition.
- **LM Studio API** for text generation (chat) and text-to-speech.
- **SNAC-based decoder** to convert TTS token streams into PCM audio.

## Features

- **Speech Recognition:** Captures audio input from the microphone and transcribes it using the Whisper model.
- **Text Generation:** Uses LM Studio’s chat API to generate responses based on the transcribed input.
- **Text-to-Speech:** Synthesizes speech from text using LM Studio’s TTS API and decodes the token stream with a SNAC-based decoder.
- **Audio Playback:** Plays the generated audio and checks its duration to warn if the audio might be truncated.

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
- [SNAC](https://github.com/hubertsiuzdak/snac) (or your local version)

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

   - Create a `config.yaml` file in the project root.
   - Populate it with your configuration details for Whisper, LM Studio API, audio settings, etc.

   Example `config.yaml`:

   ```yaml
   whisper:
     model_name: "base"
     sample_rate: 16000

   lm_studio_api:
     api_url: "http://your-api-url.com"
     chat:
       endpoint: "/v1/chat"
       model: "gemma"
       system_prompt: "You are a helpful assistant."
       max_tokens: 150
       temperature: 0.7
       top_p: 0.9
       repetition_penalty: 1.0
     tts:
       endpoint: "/v1/tts"
       model: "orpheus"
       default_voice: "default"
       max_tokens: 200
       temperature: 0.8

   tts:
     sample_rate: 24000

   audio:
     input_device: null
     output_device: null

   desired_tts_duration: 20
   ```

4. **Run the Assistant:**

   ```bash
   python morpheus_demo.py
   ```
Make sure you have your choosen LLM model and the Orpheus 4-bit GGUF loaded inside LM Studio and that you are in API mode. 

## Usage

- The assistant will begin by listening for your voice input.
- After transcribing your speech, it will generate a text response using the LM Studio API.
- The response is then converted to speech using the TTS API, decoded via SNAC, and played back.
- If the generated audio duration is below the configured threshold, a warning is printed.

## Contributing

Feel free to open issues or submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
