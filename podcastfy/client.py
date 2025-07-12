"""
Podcastfy CLI

This module provides a command-line interface for generating podcasts or transcripts
from URLs or existing transcript files. It orchestrates the content extraction,
generation, and text-to-speech conversion processes.
"""

import os
import uuid
import typer
import yaml
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.content_generator import ContentGenerator
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.config import Config, load_config
from podcastfy.utils.config_conversation import load_conversation_config
from podcastfy.utils.logger import setup_logger
from typing import List, Optional, Dict, Any
import copy

import logging

# Configure logging to show all levels and write to both file and console
""" logging.basicConfig(
    level=logging.DEBUG,  # Show all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcastfy.log'),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
) """


logger = setup_logger(__name__)

app = typer.Typer()

os.environ["LANGCHAIN_TRACING_V2"] = "False"

print("[CONFIRM] podcastfy.client imported")

__version__ = "0.1.2"


def process_content(
    urls: Optional[List[str]] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    generate_audio: bool = True,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
    output_dir: Optional[str] = None,
):
    """
    Process URLs, a transcript file, image paths, or raw text to generate a podcast or transcript.
    """
    logger.info("[process_content] Starting content processing...")
    print("[process_content] Starting content processing...")
    try:
        if config is None:
            config = load_config()

        # Load default conversation config
        conv_config = load_conversation_config()

        # Update with provided config if any
        if conversation_config:
            conv_config.configure(conversation_config)
        # Get output directories from conversation config
        tts_config = conv_config.get("text_to_speech", {})
        output_directories = tts_config.get("output_directories", {})

        if transcript_file:
            logger.info(f"Using transcript file: {transcript_file}")
            with open(transcript_file, "r") as file:
                qa_content = file.read()
        else:
            # Initialize content_extractor if needed
            content_extractor = None
            if urls or topic or (text and longform and len(text.strip()) < 100):
                content_extractor = ContentExtractor()

            content_generator = ContentGenerator(
                is_local=is_local,
                model_name=model_name,
                api_key_label=api_key_label,
                conversation_config=conv_config.to_dict()
            )

            # Always use the resolved api_key_label from ContentGenerator for further lookups
            api_key_label = content_generator.api_key_label

            combined_content = ""
            
            if urls:
                logger.info(f"Processing {len(urls)} links")
                contents = [content_extractor.extract_content(link) for link in urls]
                combined_content += "\n\n".join(contents)

            if text:
                if longform and len(text.strip()) < 100:
                    logger.info("Text too short for direct long-form generation. Extracting context...")
                    expanded_content = content_extractor.generate_topic_content(text)
                    combined_content += f"\n\n{expanded_content}"
                else:
                    combined_content += f"\n\n{text}"

            if topic:
                topic_content = content_extractor.generate_topic_content(topic)
                combined_content += f"\n\n{topic_content}"

            # Generate Q&A content using output directory from conversation config
            random_filename = f"transcript_{uuid.uuid4().hex}.txt"
            transcript_filepath = os.path.join(
                output_directories.get("transcripts", "data/transcripts"),
                random_filename,
            )
            qa_content = content_generator.generate_qa_content(
                combined_content,
                image_file_paths=image_paths or [],
                output_filepath=transcript_filepath,
                longform=longform
            )

        if generate_audio:
            logger.info(f"[process_content] Starting TTS with model: {tts_model}")
            print(f"[process_content] Starting TTS with model: {tts_model}")
            api_key = None
            if tts_model != "edge":
                api_key = getattr(config, f"{tts_model.upper().replace('MULTI', '')}_API_KEY")

            text_to_speech = TextToSpeech(
                model=tts_model,
                api_key=api_key,
                conversation_config=conv_config.to_dict(),
            )

            random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
            audio_output_dir = output_dir or "classic/agents/myagent/workspace"
            audio_file = os.path.join(audio_output_dir, random_filename)
            text_to_speech.convert_to_speech(qa_content, audio_file)
            logger.info(f"[process_content] TTS finished. Audio file: {audio_file}")
            print(f"[process_content] TTS finished. Audio file: {audio_file}")
            logger.info(f"Podcast generated successfully using {tts_model} TTS model")
            return audio_file
        else:
            logger.info(f"Transcript generated successfully: {transcript_filepath}")
            print(f"Transcript generated successfully: {transcript_filepath}")
            return transcript_filepath

    except Exception as e:
        logger.error(f"[process_content] Error: {e}")
        print(f"[process_content] Error: {e}")
        raise


@app.command()
def main(
    urls: list[str] = typer.Option(None, "--url", "-u", help="URLs to process"),
    file: typer.FileText = typer.Option(
        None, "--file", "-f", help="File containing URLs, one per line"
    ),
    transcript: typer.FileText = typer.Option(
        None, "--transcript", "-t", help="Path to a transcript file"
    ),
    tts_model: str = typer.Option(
        None,
        "--tts-model",
        "-tts",
        help="TTS model to use (openai, elevenlabs, edge, or gemini)",
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Generate only a transcript without audio"
    ),
    conversation_config_path: str = typer.Option(
        None,
        "--conversation-config",
        "-cc",
        help="Path to custom conversation configuration YAML file",
    ),
    image_paths: List[str] = typer.Option(
        None, "--image", "-i", help="Paths to image files to process"
    ),
    is_local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use a local LLM instead of a remote one (http://localhost:8080)",
    ),
    text: str = typer.Option(
        None, "--text", "-txt", help="Raw text input to be processed"
    ),
    llm_model_name: str = typer.Option(
        None, "--llm-model-name", "-m", help="LLM model name for transcript generation"
    ),
    api_key_label: str = typer.Option(
        None, "--api-key-label", "-k", help="Environment variable name for LLMAPI key"
    ),
    topic: str = typer.Option(
        None, "--topic", "-tp", help="Topic to generate podcast about"
    ),
    longform: bool = typer.Option(
        False, 
        "--longform", 
        "-lf", 
        help="Generate long-form content (only available for text input without images)"
    ),
):
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, image files, or raw text.
    """
    try:
        config = load_config()
        main_config = config.get("main", {})

        conversation_config = None
        # Load conversation config if provided
        if conversation_config_path:
            with open(conversation_config_path, "r") as f:
                conversation_config: Dict[str, Any] | None = yaml.safe_load(f)

        # Use default TTS model from conversation config if not specified
        if tts_model is None:
            tts_config = load_conversation_config().get("text_to_speech", {})
            tts_model = tts_config.get("default_tts_model", "openai")

        if transcript:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            final_output = process_content(
                transcript_file=transcript.name,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                conversation_config=conversation_config,
                config=config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )
        else:
            urls_list = urls or []
            if file:
                urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise typer.BadParameter(
                    "No input provided. Use --url, --file, --transcript, --image, --text, or --topic."
                )

            final_output = process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )

        if transcript_only:
            typer.echo(f"Transcript generated successfully: {final_output}")
        else:
            typer.echo(
                f"Podcast generated successfully using {tts_model} TTS model: {final_output}"
            )

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


def generate_podcast(
    urls: Optional[List[str]] = None,
    url_file: Optional[str] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    transcript_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    logger.info("[generate_podcast] Starting podcast generation...")
    print("[generate_podcast] Starting podcast generation...")
    result = None
    try:
        result = process_content(
            urls=urls,
            transcript_file=transcript_file,
            tts_model=tts_model,
            generate_audio=not transcript_only,
            config=config,
            conversation_config=conversation_config,
            image_paths=image_paths,
            is_local=is_local,
            text=text,
            model_name=llm_model_name,
            api_key_label=api_key_label,
            topic=topic,
            longform=longform,
            output_dir=output_dir,
        )
        logger.info(f"[generate_podcast] Podcast generation finished. Result: {result}")
        print(f"[generate_podcast] Podcast generation finished. Result: {result}")
    except Exception as e:
        logger.error(f"[generate_podcast] Error: {e}")
        print(f"[generate_podcast] Error: {e}")
        raise
    return result
