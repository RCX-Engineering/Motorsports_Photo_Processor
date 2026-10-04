# Motorsports Photo Selection and Prodessing Tool

A Python pipeline that culls a folder of motorsport panning photos, groups the
frames by driver, picks each driver's best shots, and exports tilted,
rule-of-thirds crops. It was written and is run by Claude (Anthropic's AI
coding agent), working with a photographer who shoots SCCA autocross and
track events. *Don't worry, the AI will handle almost everything for you.*

What you, the Photographer, will need to do: Install Microsoft VSCode (free) or 
another coding tool that supports AI-aided development. Any AI can walk you 
through it if you neeed help. Then point it to this git.

## Read this first: this is a starting point, not a finished tool

`motorsport_cull.py` was tuned for one photographer's camera, shooting style,
tracks, and hardware. **It is not meant to be run as-is.** Give it to an AI
coding tool (Claude Code, Cursor, Copilot, etc.) and have the tool adapt it to
your setup. For example:

> Here's a motorsport photo culling script. Read it, set it up on my machine,
> and adapt it to my photos in `<your shoot folder>`. I shoot [autocross /
> track days / rally / karting] with a [camera], and I like [tighter crops /
> no tilt / ...].

The docstring at the top of the script describes the whole pipeline, and the
named constants and comments explain each threshold, so an AI tool can tell
what to change and why.

## What it does

1. **Splits the shoot into passes** using capture timestamps. A long gap, or a
   car that suddenly looks different, starts a new pass.
2. **Finds the car** in every frame with YOLO segmentation. It then measures
   car sharpness, background pan blur and its direction, wheel spin, whether
   the car is cut off by the frame edge, exposure, and bystanders.
3. **Rejects unusable frames**: no car, car cut off, car smeared, or car much
   softer than the best frame of that pass.
4. **Identifies drivers.** EasyOCR (text recognition) reads the car number,
   and CLIP compares how the cars look when the number can't be read. An
   optional local vision-language model can read numbers instead (see below).
5. **Ranks each driver's frames.** The score combines sharpness, pan quality,
   and the LAION aesthetic predictor. It drops near-duplicates and picks a
   "hero" shot.
6. **Tilts and crops each keeper.** The car is tilted nose-up, framed with
   space ahead of the nose, and placed with the driver near a thirds point.
   Night atmosphere and track signage earn a bonus.
7. **Exports** edited JPEGs (up to 4 per driver), a `catalog.csv` that records
   every frame and decision, and a composition log. Source files are never
   modified.

It can also **learn your taste**. `--learn-from <folder of your hand-picked
photos>` fits the ranking weights to your own choices.

## Things your AI tool will likely need to change

- **Dependencies.** Install `opencv-python numpy pillow ultralytics torch
  open_clip_torch easyocr`, plus `pytest` for the tests. EasyOCR reads car
  numbers and track signage. Install the CUDA build of PyTorch if you have an
  NVIDIA GPU. The YOLO, CLIP, and aesthetic model weights download
  automatically on first run.
- **Optional: a vision model for number reading.** This is not required. It
  only adds a smarter number reader. The script can send each car crop to a
  local vision-language model served by [Ollama](https://ollama.com). If no
  Ollama server answers, it falls back to EasyOCR automatically. Use
  `--number-reader easyocr` to skip the attempt. A vision model can tell the
  competition number apart from class letters, license plates, and sponsor
  decals better than plain OCR can. We haven't gotten this path working
  reliably yet, so treat it as experimental. Any model you use needs to:
  - accept image input (listed with the "vision" capability in Ollama)
  - return structured JSON output reliably
  - read small text and digits accurately
  - fit in your GPU's memory alongside the YOLO and CLIP models

  We wrote this path against `gemma4:e4b`, but other models with these
  capabilities should also work. Point `OLLAMA_URL` and `OLLAMA_MODEL` at
  your own setup.
- **Track-specific settings.** `SIGNAGE_WORDS` (currently LANGLEY, SPEEDWAY,
  ...) is tuned for one venue. Number rules such as `NUMBER_MAX_DIGITS` follow
  SCCA conventions.
- **Style defaults.** Tilt (`--tilt`, 10°), crop fill (`--max-fill`), photos
  per driver (`--per-driver`), and the blur thresholds all reflect one
  person's taste.
- **Assumptions.** The script expects JPEGs with EXIF capture times and one
  car in view at a time, and it works best on panning shots.

## Basic usage (once adapted)

```
python motorsport_cull.py /path/to/shoot
python motorsport_cull.py /path/to/shoot --catalog-only
python motorsport_cull.py /path/to/shoot --learn-from /path/to/my_picks
python motorsport_cull.py /path/to/shoot --weights learned_weights.json
```

Output goes to `<shoot>/processed/` by default. Run `--help` to see every
option.

## Files

- `motorsport_cull.py`: the whole pipeline in one file.
- `test_composition.py`: unit tests for cropping, ranking, and parsing.
  Run them with `pytest` after making changes.

## Credits

Written by Claude (Anthropic) in Claude Code, together with the photographer
who directed it. Aesthetic scoring uses the
[LAION improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor),
detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
appearance matching uses [OpenCLIP](https://github.com/mlfoundations/open_clip),
and text reading uses [EasyOCR](https://github.com/JaidedAI/EasyOCR).
