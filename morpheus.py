#!/usr/bin/env python
import argparse
import sys
from modules.virtual_assistant import VirtualAssistant
from modules.logging import logger


def main():
    parser = argparse.ArgumentParser(description="Start the mOrpheus Virtual Assistant.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="settings.yml",
        help="Path to configuration YAML file."
    )
    args = parser.parse_args()

    try:
        assistant = VirtualAssistant(config_path=args.config)
        logger.info("Starting mOrpheus virtual assistant...")
        assistant.run()
    except KeyboardInterrupt:
        logger.info("Assistant interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("mOrpheus shutdown complete")


if __name__ == "__main__":
    main()
