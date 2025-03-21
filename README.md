# mOrpheus Virtual Assistant Demo

This project implements the **mOrpheus Virtual Assistant**, which integrates speech recognition, text generation, and text-to-speech (TTS) synthesis. The assistant utilizes the following components:

- **Whisper** for speech recognition.
- **LM Studio API** for text generation (chat) and text-to-speech.
- **SNAC-based decoder** to convert TTS token streams into PCM audio.

---

## 🚀 Features

- **Speech Recognition:** Captures audio input from the microphone and transcribes it using the Whisper model.
- **Text Generation:** Uses LM Studio’s chat API to generate responses based on the transcribed input.
- **Text-to-Speech:** Synthesizes speech from text using LM Studio’s TTS API and decodes the token stream with a SNAC-based decoder.
- **Audio Playback:** Plays the generated audio and checks its duration to warn if the audio might be truncated.

---

## 📦 Requirements

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

---

## ✅ Setup

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/morpheus-virtual-assistant.git
cd morpheus-virtual-assistant
```

2. **Create and activate a virtual environment**:

**On Linux/macOS**:

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install the dependencies**:

```bash
pip install -r requirements.txt
```

> ⚠️ Ensure your `requirements.txt` includes all required packages.  
> Example:

```txt
torch
whisper
sounddevice
scipy
numpy
requests
PyYAML
transformers
```

4. **(Optional) Find your audio input/output device IDs**:

You can use this Python snippet to list available audio devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

Look for the index numbers next to your desired microphone (input) and speaker (output), then set them in your `config.yaml`:

```yaml
audio:
  input_device: 1   # Replace with your mic ID
  output_device: 3  # Replace with your speaker ID
```

5. **Configure the application**:

Create a `config.yaml` file in the project root and populate it with your settings:

```yaml
whisper:
  model_name: "small.en"
  sample_rate: 16000

lm_studio_api:
  api_url: "http://127.0.0.1:1234"
  chat:
    endpoint: "/v1/chat/completions"
    model: "gemma-3-1b-it"
    system_prompt: "You are a smart assistant with a knack for humor."
    max_tokens: 2500
    temperature: 0.7
    top_p: 0.9
    repetition_penalty: 1.1
  tts:
    endpoint: "/v1/completions"
    model: "orpheus-3b-0.1-ft"
    default_voice: "tara"
    max_tokens: 2500
    temperature: 0.6
    top_p: 0.9
    repetition_penalty: 1.0

tts:
  sample_rate: 24000

audio:
  input_device: 15
  output_device: 21
```

6. **Run the Assistant**:

```bash
python morpheus_demo.py
```

> ✅ Make sure your chosen LLM model and the **Orpheus 4-bit GGUF** are loaded in LM Studio, and that LM Studio is in **API mode**.

---

## 🗣️ Usage

- The assistant will begin by listening for your voice input.
- After transcribing your speech, it will generate a text response using the LM Studio API.
- The response is then converted to speech using the TTS API, decoded via SNAC, and played back.
- If the generated audio duration is below the configured threshold, a warning will be shown.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests with improvements, features, or bug fixes!

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.
