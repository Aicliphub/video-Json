# YouTube Video Generation System

An advanced system for automatically generating long-form YouTube videos using AI.

## Overview

This system uses a "mastermind" architecture to coordinate multiple AI components:

1. **Script Generation**: Creates engaging scripts based on a given topic
2. **Text-to-Speech**: Converts scripts to natural-sounding audio
3. **Transcription**: Creates timestamped segments from the audio
4. **Image Prompt Generation**: Creates prompts for each segment
5. **Image Generation**: Creates visuals for each segment
6. **Video Assembly**: Combines all elements into a final video

## Architecture

| Module | Input | Output |
|--------|-------|--------|
| `mastermind.py` | Topic | Directs flow, manages state |
| `script_writer.py` | Topic | Returns script |
| `tts_generator.py` | Script | Returns TTS audio URL |
| `transcriber.py` | TTS audio URL | Returns timestamped segments |
| `prompt_generator.py` | Segment text | Returns image prompts |
| `image_generator.py` | Prompt | Returns image URL |
| `json_builder.py` | All outputs | Updates `video.json` |
| `video.py` | Final JSON | Creates video via Remotion |

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys in `.env`
4. Place reference audio in `assets/reference/`

## Usage

```bash
python -m src.mastermind --topic "Your video topic here"
```

## Configuration

Edit the `.env` file to configure API keys and other settings.

## License

MIT
