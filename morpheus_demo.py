import yaml
from modules.virtual_assistant import VirtualAssistant

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    assistant = VirtualAssistant(config)
    assistant.run()
