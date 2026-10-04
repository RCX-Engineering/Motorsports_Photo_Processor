#!/usr/bin/env python3
"""
Motorsport Photo Culling and Hero-Shot Processor
================================================

Culls a folder of motorsport panning photographs, identifies each driver,
selects each driver's best frames, and exports tilted, composed crops.

Pipeline
--------
1. Read capture times and split the shoot into passes: a gap longer than
   ``--gap`` seconds starts a new pass, and after analysis so does a shorter
   gap where the car looks different. One car is assumed in view at a time.
2. Analyze every frame: segment the vehicle, then measure
     * vehicle sharpness (Laplacian variance inside the vehicle mask),
     * background sharpness and the vehicle-to-background blur ratio,
     * background blur coherence (a clean pan blurs in one direction),
     * vehicle size, frame-edge contact, clipped exposure, and bystanders,
     * wheel spin blur and body attitude from detected wheels,
     * background drift from the previous frame of the same pass.
3. Reject unusable frames: no vehicle, vehicle cut off by the frame edge,
   car smeared against a sharp background, car unusably blurry, or vehicle
   soft compared with the sharpest frame of the pass. Photos named in ``keep.txt`` in the shoot
   folder skip these tests. Small
   vehicles are kept; the photographer framed them that way on purpose.
4. For surviving frames, score aesthetics (LAION aesthetic predictor on
   CLIP ViT-L/14), read the car number (a vision-language model served by
   Ollama, or EasyOCR), and embed the car's appearance (CLIP).
5. Merge passes into drivers: passes reading the same car number are one
   driver; passes without a readable number join the most similar-looking
   pass above ``--match-threshold``. Passes with different numbers are never
   merged.
6. Drop blurry frames (at source or display size) when the same driver has
   a sharp one, keeping each driver's best if none is sharp; then drop
   slightly soft frames when the driver has a crisp frame of the same view
   (facing, side-on or angled). Rank each driver's frames (or each pass's, with ``--select-by pass``)
   with a weighted score whose features are normalized within the group.
   The top frame is the hero; every frame that passed rejection is a keeper.
7. For every keeper, decide once which way the car faces (roofline offset,
   then background drift, then ``--default-direction``). That one value sets
   both the tilt sign (nose-up) and the side the lead space goes on.
8. Tilt, then compose a tight crop in the rotated frame: car as large as
   fits (up to ``--max-fill`` of the width), tail close to the frame edge,
   open space ahead of the nose, driver as near a thirds intersection as the
   tight crop allows. A looser crop
   is used only when the aesthetic predictor rates it clearly better, and a
   centered crop is the fallback when no thirds layout fits in the image.
9. Score each crop's coolness (night atmosphere, racetrack signage, and
   aesthetics), rank each driver's photos, and export up to ``--per-driver``
   of them, skipping near-identical shots.
10. Write one JPEG per exported photo (heroes marked ``_hero``), a
   composition log, and a CSV catalog of every frame and decision.

Learning your taste
-------------------
``--learn-from PICKS`` points at a folder of the photos you chose by hand
from the same shoot. Picks are matched to the originals by capture time or
file name, ranking weights are fitted to your choices, agreement is reported,
and the weights are saved as ``learned_weights.json``. Later runs apply them
with ``--weights learned_weights.json``.

Source files are never moved, renamed, modified, or deleted. Frames whose
processed output already exists are skipped unless ``--force`` is given.

Usage
-----
    python motorsport_cull.py /path/to/shoot
    python motorsport_cull.py /path/to/shoot --tilt 15 --gap 12
    python motorsport_cull.py /path/to/shoot --catalog-only
    python motorsport_cull.py /path/to/shoot --learn-from /path/to/my_picks
    python motorsport_cull.py /path/to/shoot --weights learned_weights.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JPEG_EXTENSIONS = {".jpg", ".jpeg"}

# COCO class IDs used by the pretrained YOLO segmentation models.
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CAR_CLASS = 2

# Competition cars are detected as cars; trucks and buses in frame are usually
# parked support vehicles. A car at least CAR_PREFERENCE_RATIO of the largest
# vehicle's area is chosen over a larger truck or bus.
CAR_PREFERENCE_RATIO = 0.25

# EXIF tag IDs.
EXIF_IFD_POINTER = 0x8769
TAG_ORIENTATION = 0x0112
TAG_DATETIME = 0x0132
TAG_DATETIME_ORIGINAL = 0x9003
TAG_SUBSEC_ORIGINAL = 0x9291

# Long edge, in pixels, of the working copy used for analysis and crop
# planning. Sharpness is always measured at this resolution so values are
# comparable across frames from the same camera.
ANALYSIS_LONG_EDGE = 1600

# Long edge of the grayscale thumbnail used for background-drift estimation.
DRIFT_LONG_EDGE = 480

# Longest interval, in seconds, between frames whose background drift is
# measured. Beyond a burst, the pan moves the background too far between
# frames and phase correlation locks onto unrelated structure.
MAX_DRIFT_INTERVAL = 0.25

# Passes: a new pass starts after a gap longer than --gap seconds, or after
# a gap longer than SPLIT_MIN_GAP seconds when the car looks different (CLIP
# similarity of the vehicle crops below SPLIT_SIMILARITY). At busy events
# cars follow each other closely, so time alone would merge several cars.
# A front-to-rear view change can also split one car's pass; such pieces are
# merged again by car number or appearance when drivers are identified.
SPLIT_MIN_GAP = 1.5
SPLIT_SIMILARITY = 0.78

# Minimum phase-correlation peak (0 to 1) for a drift measurement to count.
MIN_DRIFT_RESPONSE = 0.03

# Minimum background drift, in thumbnail pixels, for a direction vote to count.
MIN_DRIFT_PIXELS = 2.0

# Blur scores: the vehicle is blurred again by BLUR_KERNEL pixels
# horizontally and vertically. A sharp vehicle loses most of its edge detail;
# an already blurry one barely changes. The score is the fraction of detail
# that survives, on the worse axis: about 0.3 for a crisp pan, above 0.5 for
# missed focus or a pan that did not track the car. It is measured three ways:
#   blur          the car from the full-resolution image, reduced to at most
#                 BLUR_CROP_WIDTH pixels wide;
#   display_blur  the car as a viewer sees it: resized to its width when a
#                 crop at the default fill is shown VIEW_WIDTH pixels wide, so
#                 distant cars enlarged by a tight crop must be sharper at the
#                 source. Sensor noise (heavy at night) looks like fine detail
#                 and would hide blur, so the car is first smoothed by
#                 DISPLAY_DENOISE_SIGMA pixels and only its strongest
#                 DISPLAY_STRONG_EDGES share of edges (lettering, body lines)
#                 is judged. This score runs higher: about 0.5 to 0.6 is crisp,
#                 above 0.7 visibly smeared;
#   car/bg blur   the car and the background at the analysis size, so a car
#                 smeared against a sharp background (a missed pan) stands out.
BLUR_CROP_WIDTH = 768
BLUR_KERNEL = 9
VIEW_WIDTH = 2048
DISPLAY_DENOISE_SIGMA = 1.0
DISPLAY_STRONG_EDGES = 0.10

# Always rejected, even as a driver's only photo: a missed pan (car blur at
# least MISSED_PAN_GAP above the background's and at least MISSED_PAN_BLUR
# itself) or a car smeared beyond UNUSABLE_BLUR.
MISSED_PAN_GAP = 0.17
MISSED_PAN_BLUR = 0.44
UNUSABLE_BLUR = 0.55

# Rejected only when the driver has a photo without these problems: blur
# above --max-blur, display_blur above --max-display-blur, or a car-to-
# background gap above SOFT_PAN_GAP. A driver with no such photo keeps the
# one with the lowest display_blur.
SOFT_PAN_GAP = 0.12

# Soft duplicates: a frame with blur above CRISP_BLUR is dropped when the same
# driver has a crisp frame of the same view, since that shot is already
# covered better. The view is the facing plus side-on (vehicle box at least
# SIDE_VIEW_ASPECT times wider than tall) or angled. Soft frames that show a
# view no crisp frame covers are kept; they are the best available of it.
CRISP_BLUR = 0.35
SIDE_VIEW_ASPECT = 2.4

# Photos listed in this file (one file name or stem per line, # comments
# allowed) in the shoot folder are kept regardless of blur, softness, or
# duplication. Use it for shots whose composition earns a place.
KEEP_LIST_NAME = "keep.txt"

# Width of the frame border band, in analysis pixels, checked for vehicle
# contact, and the number of vehicle pixels in that band that marks the
# vehicle as cut off.
EDGE_BAND_PIXELS = 2
EDGE_CONTACT_PIXELS = 8

# Minimum bystander size as a fraction of the frame area.
MIN_PERSON_AREA = 0.003

# Crop planning: padding kept around the vehicle (fraction of its size on each
# side), number of crop sizes tried, and the composition error (fraction of
# the crop size) accepted without searching smaller crops.
SUBJECT_PADDING = 0.06
CROP_SEARCH_STEPS = 16
ANCHOR_TOLERANCE = 0.02

# Facing: in side and three-quarter views the cabin sits behind the middle of
# the car, so the roofline (vehicle columns whose top lies in the upper
# ROOF_BAND of the vehicle box) is offset toward the tail. Offsets smaller
# than FACING_MIN_OFFSET (fraction of box width) are inconclusive, as in
# head-on and tail-on views.
ROOF_BAND = 0.25
FACING_MIN_OFFSET = 0.02

# When the roofline is inconclusive (near head-on views), the taller end of
# the vehicle is taken as the tail: the flank and pillars on the tail side
# stand taller than the tapering nose. END_SLICE is the fraction of the box
# width compared at each end; differences below END_MIN_OFFSET (fraction of
# box height) are inconclusive.
END_SLICE = 0.2
END_MIN_OFFSET = 0.03

# Driver: the most confident person detected inside the upper DRIVER_MAX_DEPTH
# of the vehicle box, at DRIVER_MIN_CONFIDENCE or above, is the driver; the
# helmet sits HEAD_DEPTH down that person box. Without a detection, the
# cockpit is estimated at the box's horizontal center and COCKPIT_DEPTH down
# from the roof, which is where detected drivers land on average.
DRIVER_MIN_CONFIDENCE = 0.25
DRIVER_MAX_DEPTH = 0.6
HEAD_DEPTH = 0.3
COCKPIT_DEPTH = 0.23

# Nose and tail points are the centroids of the front and rear NOSE_SLICE of
# the vehicle's width.
NOSE_SLICE = 0.12

# Horizon: long background lines within HORIZON_MAX_ANGLE degrees of level
# and at least HORIZON_MIN_LENGTH of the frame width (walls, track edges).
HORIZON_MAX_ANGLE = 20.0
HORIZON_MIN_LENGTH = 0.2

# Composition candidates: tilt magnitudes as multiples of --tilt and the
# horizontal third lines the driver is steered toward (fraction of height).
# The nose must end at least MIN_NOSE_RISE degrees above the tail, which may
# raise the tilt up to MAX_TILT on shots where perspective drops the nose.
TILT_FACTORS = (0.6, 1.0, 1.4)
DRIVER_ROWS = (1 / 3, 2 / 3)
MIN_NOSE_RISE = 2.0
MAX_TILT = 35.0

# Tight crops: the vehicle fills FILL_LEVELS of the crop width, tightest
# first, and the tightest level that fits is used, up to --max-fill
# (default DEFAULT_MAX_FILL).
# Behind the tail only TAIL_MARGIN of the width is left; ahead of the nose at
# least LEAD_MIN stays open; above and below the car at least VERTICAL_MARGIN
# of the height. Within those limits the crop is placed to bring the driver
# as close as possible to a thirds intersection; THIRDS_TOLERANCE is the
# distance (fraction of the crop) at which that preference scores zero.
FILL_LEVELS = (0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.66, 0.60)
DEFAULT_MAX_FILL = 0.84
TAIL_MARGIN = 0.03
LEAD_MIN = 0.08
VERTICAL_MARGIN = 0.04
THIRDS_TOLERANCE = 0.2

# A looser fill level replaces the tight crop only when the aesthetic
# predictor rates it at least ZOOM_OUT_MARGIN higher, e.g. for an interesting
# foreground or background.
ZOOM_OUT_MARGIN = 0.3

# Coolness of a crop, added to its aesthetic rating when choosing the zoom
# and used again when ranking a driver's photos:
#   night    dark surroundings (luminance below NIGHT_DARK_LEVEL) filling up
#            to NIGHT_DARK_TARGET of the crop, counted only when the car
#            itself is lit (90th-percentile luminance CAR_LIT_TARGET or more);
#   signage  how much of the racetrack signage found in the full frame (words
#            close to SIGNAGE_WORDS, e.g. LANGLEY SPEEDWAY) the crop contains.
NIGHT_DARK_LEVEL = 0.10
NIGHT_DARK_TARGET = 0.25
CAR_LIT_TARGET = 0.5
NIGHT_BONUS = 0.4
SIGN_BONUS = 0.6
SIGNAGE_WORDS = ("LANGLEY", "SPEEDWAY", "RACEWAY", "MOTORSPORTS", "NASCAR")
SIGNAGE_MIN_CONFIDENCE = 0.3
SIGNAGE_MATCH = 0.75

# Selection: at most PHOTOS_PER_DRIVER photos per driver (--per-driver), best
# first, skipping any whose crop looks at least DUPLICATE_SIMILARITY alike
# (CLIP embedding cosine) to one already chosen.
PHOTOS_PER_DRIVER = 4
DUPLICATE_SIMILARITY = 0.94

# Score weights for choosing among candidates of the same fill level, and the
# vehicle width (fraction of crop width) used by the centered fallback.
COMPOSE_WEIGHTS = {
    "thirds": 1.0,      # driver near the intersection
    "lead": 0.3,        # open space ahead of the nose, up to a full third
    "horizon": 0.5,     # background tilt close to --tilt, nose-up
}
CENTER_SUBJECT_WIDTH = 0.7

# Wheel search: radius range as a fraction of vehicle box height, the lowest
# point (fraction of box height from the top) a wheel center may sit above,
# and the Hough "perfectness" threshold (0 to 1; higher finds fewer circles).
WHEEL_RADIUS_RANGE = (0.12, 0.40)
WHEEL_CENTER_MIN_DEPTH = 0.35
WHEEL_CIRCLE_PERFECTNESS = 0.75

# Annulus inside each wheel used for spin blur, as fractions of its radius:
# covers spokes and rim face, excludes hub and tire sidewall.
WHEEL_ANNULUS = (0.20, 0.75)

# Car-number reading with EasyOCR: one to three digits, optionally followed by
# an SCCA class (e.g. "27 GST"), text height at least this fraction of the
# vehicle crop height (rejects sponsor text), and the long edge the vehicle
# crop is limited to before reading. Letters are read so the class is not
# misread as digits, then dropped.
NUMBER_PATTERN = re.compile(r"(\d{1,3})[A-Z]{0,4}")
NUMBER_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER_MIN_HEIGHT = 0.06
NUMBER_CROP_LONG_EDGE = 1600

# Minimum OCR confidence for a read to count. Clean door numbers read at
# 0.9 or higher; stylized decals misread as extra digits score far lower.
NUMBER_MIN_CONFIDENCE = 0.5

# A pass's number is accepted when its summed reading score reaches
# NUMBER_MIN_SCORE and beats the runner-up by NUMBER_MARGIN times. A clear
# read of a typical door number scores about 0.1 to 0.15 per frame.
NUMBER_MIN_SCORE = 0.10
NUMBER_MARGIN = 1.5

# SCCA competition numbers have at most this many digits; longer reads are
# license plates or class letters misread as digits.
NUMBER_MAX_DIGITS = 3

# Vision-language number reading: a model served by Ollama is shown the
# vehicle crop (long edge VLM_IMAGE_EDGE) and asked for the competition
# number. Its stated confidence becomes the read's score; "low" reads are
# dropped. The model is unloaded OLLAMA_KEEP_ALIVE after the last read so
# it does not hold video memory other tools need. If the server does not
# answer within OLLAMA_CONNECT_TIMEOUT seconds, EasyOCR is used instead.
OLLAMA_URL = "http://127.0.0.1:11435"
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_KEEP_ALIVE = "2m"
OLLAMA_CONNECT_TIMEOUT = 10
OLLAMA_READ_TIMEOUT = 300
OLLAMA_RETRY_WAITS = (5, 15)   # seconds to wait before each retry of a failed read
VLM_IMAGE_EDGE = 1024
VLM_CONFIDENCE_SCORES = {"high": 1.0, "medium": 0.6}
VLM_NUMBER_PROMPT = (
    "Read this autocross car's competition number: the large digits on its door, "
    "side, hood, or windshield number panel. A class designation in letters often "
    "sits before, after, or under the number (e.g. a class followed by digits); "
    "report those letters separately as the class, never as part of the number, and "
    "never read letters as digits. Ignore license plates, sponsor decals, and "
    "driver names. Report only a number you can actually see in this image; if no "
    "competition number is visible or readable, the number must be null. Reply as "
    'JSON: {"number": string or null, "class": string or null, '
    '"confidence": "high" or "medium" or "low"}'
)

# CLIP backbone shared by the aesthetic predictor and appearance matching,
# and the LAION aesthetic predictor head trained on its embeddings.
CLIP_ARCHITECTURE = "ViT-L-14-quickgelu"  # OpenAI weights were trained with QuickGELU
CLIP_PRETRAINED = "openai"
AESTHETIC_HEAD_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
                      "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth")
CACHE_DIR = Path.home() / ".cache" / "motorsport_cull"

# Hugging Face repository open_clip downloads the CLIP weights from, as named
# in the Hugging Face cache.
CLIP_HF_CACHE_NAME = "models--timm--vit_large_patch14_clip_224.openai"

# Ranking weights. Each feature is min-max normalized within its ranking
# group (driver or pass) before weighting, so weights express relative
# importance. Negative weights are penalties. A feature that could not be
# measured on a frame is treated as the middle of its group's range.
RANKING_WEIGHTS = {
    "car_sharpness": 0.15,   # the vehicle itself is crisp
    "blur": -0.15,           # detail that survives re-blurring (lower is sharper)
    "log_blur_ratio": 0.20,  # vehicle crisp relative to background (good pan)
    "pan_coherence": 0.08,   # background blur runs in a single direction
    "subject_area": 0.12,    # vehicle fills more of the frame
    "wheel_blur": 0.10,      # wheels visibly spinning
    "attitude": 0.05,        # body pitched or rolled relative to the wheels
    "aesthetic": 0.15,       # LAION aesthetic predictor score of the chosen crop
    "night": 0.10,           # dark night surroundings around a lit car
    "signage": 0.10,         # racetrack signage in the crop
    "clip_fraction": -0.05,  # blown highlights or crushed shadows on the vehicle
    "people": -0.05,         # bystanders or workers in frame
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One object detected by the segmentation model, in analysis pixels."""

    class_id: int
    confidence: float
    box: np.ndarray       # x0, y0, x1, y1
    polygon: np.ndarray   # (N, 2) outline; box corners when no mask exists

    @property
    def area(self) -> float:
        return float(abs(cv2.contourArea(self.polygon.astype(np.float32))))


@dataclass
class Frame:
    """Everything measured and decided about one photograph."""

    path: Path
    capture_time: float
    time_source: str
    group: int = -1
    width: int = 0
    height: int = 0
    analysis_scale: float = 1.0                 # full pixels per analysis pixel
    analysis_size: tuple[int, int] = (0, 0)     # analysis (width, height)
    subject_class: str = ""
    subject_confidence: float = 0.0
    subject_polygon: np.ndarray | None = None
    subject_area: float = 0.0
    touches_edge: bool = False
    car_sharpness: float = 0.0
    blur: float = float("nan")
    display_blur: float = float("nan")
    car_blur_small: float = float("nan")         # analysis size, for the pan gap
    bg_blur_small: float = float("nan")
    quality: str = ""                           # "good" or "best available"
    signs: list = field(default_factory=list)   # signage word boxes, analysis pixels
    car_lit: float = float("nan")
    crop_aesthetic: float = float("nan")
    night: float = float("nan")
    signage: float = float("nan")
    crop_embedding: np.ndarray | None = None
    composition: Composition | None = None
    bg_sharpness: float = float("nan")
    blur_ratio: float = 1.0
    pan_coherence: float = 0.0
    clip_fraction: float = 0.0
    people: int = 0
    wheel_count: int = 0
    wheel_blur: float = float("nan")
    attitude: float = float("nan")
    aesthetic: float = float("nan")
    car_number: str = ""
    number_score: float = 0.0
    embedding: np.ndarray | None = None
    car_embedding: np.ndarray | None = None     # every frame's car, for pass splitting
    driver: str = ""
    picked: bool = False
    keep: bool = False                          # listed in keep.txt
    drift_x: float = float("nan")
    drift_response: float = 0.0
    roof_offset: float = float("nan")
    end_offset: float = float("nan")
    facing: str = ""                          # "left" or "right"
    facing_source: str = ""
    driver_point: tuple[float, float] | None = None   # analysis pixels
    driver_source: str = ""
    horizon: float = float("nan")               # degrees, counterclockwise positive
    compose_rule: str = ""
    rotation: float = float("nan")              # degrees applied, counterclockwise positive
    driver_frac: tuple[float, float] = (float("nan"), float("nan"))
    compose_score: float = float("nan")
    compose_candidates: int = 0
    status: str = "pending"
    reason: str = ""
    rank: int = 0
    score: float = float("nan")
    outputs: list[str] = field(default_factory=list)


@dataclass
class CropPlan:
    """A tilt angle and crop rectangle in rotated analysis-canvas pixels."""

    angle: float
    x: int
    y: int
    width: int
    height: int
    canvas_width: int
    canvas_height: int


# ---------------------------------------------------------------------------
# Image input and output
# ---------------------------------------------------------------------------

def read_capture_time(path: Path) -> tuple[float, str]:
    """Return (POSIX timestamp, source) from EXIF, falling back to file mtime."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(EXIF_IFD_POINTER)
            stamp = exif_ifd.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
            subsec = exif_ifd.get(TAG_SUBSEC_ORIGINAL)
        if stamp:
            text = str(stamp).strip("\x00 ")
            value = datetime.strptime(text, "%Y:%m:%d %H:%M:%S").timestamp()
            digits = "".join(ch for ch in str(subsec or "") if ch.isdigit())
            if digits:
                value += float("0." + digits)
            return value, "exif"
    except (OSError, ValueError):
        pass
    return path.stat().st_mtime, "file"


def load_upright(path: Path) -> tuple[np.ndarray, Image.Exif, bytes | None]:
    """Load a JPEG with EXIF orientation applied, as an RGB array."""
    with Image.open(path) as img:
        exif = img.getexif()
        icc_profile = img.info.get("icc_profile")
        upright = ImageOps.exif_transpose(img)
        rgb = np.asarray(upright.convert("RGB"))
    return rgb, exif, icc_profile


def save_jpeg(rgb: np.ndarray, path: Path, exif: Image.Exif,
              icc_profile: bytes | None, quality: int) -> None:
    """Save an RGB array as JPEG, keeping camera metadata and color profile.

    Pixels are already upright, so the orientation tag is reset to normal.
    """
    exif[TAG_ORIENTATION] = 1
    options = {"quality": quality, "subsampling": 0, "exif": exif.tobytes()}
    if icc_profile:
        options["icc_profile"] = icc_profile
    Image.fromarray(np.ascontiguousarray(rgb)).save(path, "JPEG", **options)


def downscale(image: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    """Shrink an image so its long edge is at most ``long_edge`` pixels."""
    height, width = image.shape[:2]
    factor = min(1.0, long_edge / max(height, width))
    if factor >= 1.0:
        return image.copy(), 1.0
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), factor


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class VehicleDetector:
    """Wraps an Ultralytics YOLO segmentation model for vehicles and people."""

    def __init__(self, weights: str, confidence: float, image_size: int,
                 device: str | None) -> None:
        from ultralytics import YOLO  # imported here so --help works without it

        self.model = YOLO(weights)
        self.confidence = confidence
        # Predict at the lower driver threshold; callers apply ``confidence``
        # to vehicles and bystanders.
        self.options = {
            "conf": min(confidence, DRIVER_MIN_CONFIDENCE),
            "imgsz": image_size,
            "classes": [PERSON_CLASS, *VEHICLE_CLASSES],
            "verbose": False,
        }
        if device:
            self.options["device"] = device

    def __call__(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # Ultralytics expects BGR
        result = self.model.predict(bgr, **self.options)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        outlines = result.masks.xy if result.masks is not None else [None] * len(boxes)

        detections = []
        for class_id, confidence, box, outline in zip(class_ids, confidences, boxes, outlines):
            if outline is None or len(outline) < 3:
                x0, y0, x1, y1 = box
                outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            detections.append(Detection(int(class_id), float(confidence), box,
                                        np.asarray(outline, dtype=np.float32)))
        return detections


def polygon_mask(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize an outline into a 0/255 mask."""
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


# ---------------------------------------------------------------------------
# Frame measurements
# ---------------------------------------------------------------------------

def measure_subject(frame: Frame, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Measure sharpness, pan quality, framing, and exposure of the vehicle."""
    height, width = mask.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    # Erode the vehicle mask so its silhouette edge against a blurred
    # background does not inflate vehicle sharpness; dilate it so vehicle
    # pixels do not leak into the background measurement.
    size = max(3, int(0.01 * max(height, width)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    inner = cv2.erode(mask, kernel) > 0
    if np.count_nonzero(inner) < 50:
        inner = mask > 0
    background = cv2.dilate(mask, kernel, iterations=2) == 0

    frame.subject_area = float(np.count_nonzero(mask)) / mask.size
    frame.car_sharpness = float(laplacian[inner].var())

    if np.count_nonzero(background) > 50:
        frame.bg_sharpness = float(laplacian[background].var())
        frame.blur_ratio = frame.car_sharpness / (frame.bg_sharpness + 1e-6)

        # Structure-tensor coherence of background gradients: near 1 when the
        # background is streaked in one direction (clean pan), near 0 when it
        # is blurred evenly in all directions (shake or missed focus).
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[background]
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[background]
        jxx, jyy, jxy = float(np.dot(gx, gx)), float(np.dot(gy, gy)), float(np.dot(gx, gy))
        trace = jxx + jyy
        if trace > 0:
            frame.pan_coherence = math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / trace

    subject_pixels = rgb[inner]
    clipped = (subject_pixels.max(axis=1) >= 250) | (subject_pixels.max(axis=1) <= 5)
    frame.clip_fraction = float(clipped.mean()) if clipped.size else 0.0

    band = EDGE_BAND_PIXELS
    border = np.concatenate([mask[:band].ravel(), mask[-band:].ravel(),
                             mask[:, :band].ravel(), mask[:, -band:].ravel()])
    frame.touches_edge = int(np.count_nonzero(border)) > EDGE_CONTACT_PIXELS


def find_wheels(gray: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float]]:
    """Locate up to two wheels as circles (x, y, radius) in analysis pixels.

    Tire outlines stay circular even when the wheel is spinning, so a circle
    search finds them in side and near-side views. Strongly oblique views make
    wheels elliptical; those frames return no wheels and the wheel features
    are treated as unmeasured.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    box_height = y1 - y0
    min_radius = max(6, int(WHEEL_RADIUS_RANGE[0] * box_height))
    max_radius = max(min_radius + 2, int(WHEEL_RADIUS_RANGE[1] * box_height))

    region = np.clip(gray[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                               minDist=2 * min_radius, param1=300,
                               param2=WHEEL_CIRCLE_PERFECTNESS,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []

    found = []
    for cx, cy, radius in circles[0]:
        x, y = cx + x0, cy + y0
        inside = mask[min(int(y), mask.shape[0] - 1), min(int(x), mask.shape[1] - 1)] > 0
        if inside and y >= y0 + WHEEL_CENTER_MIN_DEPTH * box_height:
            found.append((float(x), float(y), float(radius)))
    if len(found) < 2:
        return found[:1]

    # Prefer the widest-spaced pair of similar size and height: front and rear.
    best_pair, best_span = None, 0.0
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            (xa, ya, ra), (xb, yb, rb) = found[i], found[j]
            span = abs(xb - xa)
            similar = 0.75 <= ra / rb <= 1.33
            level = abs(yb - ya) <= 0.35 * span
            if similar and level and span >= 2.2 * (ra + rb) / 2 and span > best_span:
                best_pair, best_span = (found[i], found[j]), span
    return sorted(best_pair) if best_pair else found[:1]


def wheel_spin_blur(gray: np.ndarray, wheels: list[tuple[float, float, float]]) -> float:
    """Fraction of gradient energy that is radial inside the wheel faces.

    Spokes of a still wheel produce edges across the tangential direction;
    rotation smears them away and leaves mostly circular edges, whose
    gradients point radially. Values near 1 indicate a spinning wheel.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    radial_energy = total_energy = 0.0
    height, width = gray.shape
    for cx, cy, radius in wheels:
        x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)
        y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)
        ring = (distance >= WHEEL_ANNULUS[0] * radius) & (distance <= WHEEL_ANNULUS[1] * radius)
        if not np.any(ring):
            continue
        ux, uy = dx[ring] / distance[ring], dy[ring] / distance[ring]
        wx, wy = gx[y0:y1, x0:x1][ring], gy[y0:y1, x0:x1][ring]
        radial_energy += float(np.sum((wx * ux + wy * uy) ** 2))
        total_energy += float(np.sum(wx ** 2 + wy ** 2))
    return radial_energy / total_energy if total_energy > 0 else float("nan")


def body_attitude(mask: np.ndarray, wheels: list[tuple[float, float, float]]) -> float:
    """Angle in degrees between the body's lower edge and the wheel-center line.

    Between the wheels the lowest vehicle pixel in each column traces the
    rocker line. Its angle relative to the line joining the wheel centers
    rises with dive, squat, or roll as seen from the camera. A car's static
    rake is included, but it is constant for one car, so it cancels when a
    driver's frames are ranked against each other.
    """
    if len(wheels) != 2:
        return float("nan")
    (xa, ya, ra), (xb, yb, rb) = wheels
    start, stop = int(xa + 1.1 * ra), int(xb - 1.1 * rb)
    if stop - start < 10:
        return float("nan")
    columns = mask[:, start:stop] > 0
    has_pixels = columns.any(axis=0)
    if np.count_nonzero(has_pixels) < 10:
        return float("nan")
    lowest = columns.shape[0] - 1 - np.argmax(columns[::-1], axis=0)
    x = np.arange(start, stop)[has_pixels]
    slope = np.polyfit(x, lowest[has_pixels].astype(np.float64), 1)[0]
    body_angle = math.degrees(math.atan(slope))
    wheel_angle = math.degrees(math.atan2(yb - ya, xb - xa))
    return abs(body_angle - wheel_angle)


def reblur_score(gray: np.ndarray, region: np.ndarray, strongest: float = 1.0) -> float:
    """Fraction of edge detail in ``region`` that survives an extra blur, worse axis.

    No-reference blur measure (Crete-Roffet et al., 2007). It compares
    neighbor differences before and after a box blur along each axis, so it
    depends on how blurred the edges already are rather than on how much
    texture the vehicle has. With ``strongest`` below 1, only that share of
    the largest differences is counted, ignoring faint noise and texture.
    """
    scores = []
    for axis, ksize in ((1, (BLUR_KERNEL, 1)), (0, (1, BLUR_KERNEL))):
        blurred = cv2.blur(gray, ksize)
        detail = np.abs(np.diff(gray, axis=axis))
        remaining = np.abs(np.diff(blurred, axis=axis))
        inside = region[:, 1:] if axis == 1 else region[1:, :]
        if strongest < 1.0 and np.count_nonzero(inside) >= 50:
            inside = inside & (detail >= np.percentile(detail[inside], 100 * (1 - strongest)))
        total = float(detail[inside].sum())
        lost = float(np.maximum(0.0, detail - remaining)[inside].sum())
        scores.append(1.0 - lost / total if total > 0 else 1.0)
    return max(scores)


def measure_blur(frame: Frame, rgb: np.ndarray) -> None:
    """Blur score of the vehicle, measured on the full-resolution image.

    Downscaling hides blur, so small vehicles are measured on their own
    pixels rather than on the analysis copy.
    """
    outline = frame.subject_polygon * frame.analysis_scale
    height, width = rgb.shape[:2]
    x0 = max(0, int(outline[:, 0].min()))
    y0 = max(0, int(outline[:, 1].min()))
    x1 = min(width, int(math.ceil(outline[:, 0].max())))
    y1 = min(height, int(math.ceil(outline[:, 1].max())))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    factor = min(1.0, BLUR_CROP_WIDTH / gray.shape[1])
    if factor < 1.0:
        size = (max(1, round(gray.shape[1] * factor)), max(1, round(gray.shape[0] * factor)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    mask = polygon_mask((outline - [x0, y0]) * factor, gray.shape[1], gray.shape[0])
    region = cv2.erode(mask, np.ones((BLUR_KERNEL, BLUR_KERNEL), np.uint8)) > 0
    if np.count_nonzero(region) < 50:
        region = mask > 0
    frame.blur = reblur_score(gray, region)

    frame.display_blur = display_blur_score(rgb[y0:y1, x0:x1], outline - [x0, y0])


def display_blur_score(car_rgb: np.ndarray, outline: np.ndarray,
                       fill: float = DEFAULT_MAX_FILL) -> float:
    """Blur of a car as seen in a crop at ``fill`` shown VIEW_WIDTH pixels wide.

    ``outline`` is the car's polygon in ``car_rgb`` pixels. The car is
    resized to its viewed width (enlarging distant cars, as the crop does),
    lightly smoothed to remove sensor noise, and judged on its strongest edges.
    """
    gray = cv2.cvtColor(car_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    view_width = int(round(fill * VIEW_WIDTH))
    factor = view_width / gray.shape[1]
    size = (view_width, max(1, round(gray.shape[0] * factor)))
    gray = cv2.resize(gray, size,
                      interpolation=cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), DISPLAY_DENOISE_SIGMA)
    mask = polygon_mask(outline * factor, size[0], size[1])
    region = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
    if np.count_nonzero(region) < 50:
        region = mask > 0
    return reblur_score(gray, region, DISPLAY_STRONG_EDGES)


def measure_pan_blur(frame: Frame, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Blur of the car and of the background at the same (analysis) scale."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    car = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    background = cv2.dilate(mask, np.ones((25, 25), np.uint8)) == 0
    if np.count_nonzero(car) >= 50 and np.count_nonzero(background) >= 50:
        frame.car_blur_small = reblur_score(gray, car)
        frame.bg_blur_small = reblur_score(gray, background)


def pan_gap(frame: Frame) -> float:
    """How much blurrier the car is than its background (NaN if unmeasured)."""
    return frame.car_blur_small - frame.bg_blur_small


def hard_blur_reason(frame: Frame) -> str:
    """Why a frame is too blurry to keep under any circumstances, or ""."""
    gap = pan_gap(frame)
    if math.isfinite(gap) and gap >= MISSED_PAN_GAP and frame.car_blur_small >= MISSED_PAN_BLUR:
        return f"car blurred against a sharp background (gap {gap:.2f})"
    if math.isfinite(frame.blur) and frame.blur > UNUSABLE_BLUR:
        return f"vehicle unusably blurry (blur {frame.blur:.2f})"
    return ""


def weak_reasons(frame: Frame, args: argparse.Namespace) -> list[str]:
    """Blur problems that disqualify a frame when the driver has a better one."""
    reasons = []
    if math.isfinite(frame.blur) and frame.blur > args.max_blur:
        reasons.append(f"blur {frame.blur:.2f}")
    if math.isfinite(frame.display_blur) and frame.display_blur > args.max_display_blur:
        reasons.append(f"blurry at display size {frame.display_blur:.2f}")
    gap = pan_gap(frame)
    if math.isfinite(gap) and gap > SOFT_PAN_GAP:
        reasons.append(f"car blurrier than background by {gap:.2f}")
    return reasons


def measure_wheels(frame: Frame, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Detect wheels and record spin blur and body attitude."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    wheels = find_wheels(gray, mask)
    frame.wheel_count = len(wheels)
    if wheels:
        frame.wheel_blur = wheel_spin_blur(gray, wheels)
        frame.attitude = body_attitude(mask, wheels)


def roof_offset(mask: np.ndarray) -> float:
    """Horizontal offset of the roofline from the vehicle box center.

    Returns a fraction of the box width: positive when the roof sits right of
    center (tail on the right, so the car faces left), negative when it sits
    left of center (car faces right). NaN when there is no vehicle.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    columns = mask[:, x0:x1] > 0
    tops = np.where(columns.any(axis=0), np.argmax(columns, axis=0), mask.shape[0])
    roof = np.flatnonzero(tops <= y0 + ROOF_BAND * (y1 - y0)) + x0
    if roof.size == 0:
        return float("nan")
    return float((roof.mean() - (x0 + x1 - 1) / 2) / (x1 - x0))


def end_offset(mask: np.ndarray) -> float:
    """Height of the vehicle's right end minus its left end, as a fraction of box height.

    Same sign convention as ``roof_offset``: positive when the taller end
    (the tail) is on the right, so the car faces left.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    heights = np.count_nonzero(mask[:, x0:x1], axis=0)
    band = max(1, int(END_SLICE * (x1 - x0)))
    return float((heights[-band:].mean() - heights[:band].mean()) / (y1 - y0))


def resolve_facing(frame: Frame, args: argparse.Namespace) -> None:
    """Decide which way the vehicle faces, once, from the strongest available cue.

    The roofline offset decides side and three-quarter views; the taller end
    decides most near head-on views. Beyond that, background drift between
    burst frames (the background drifts opposite to the pan), then
    ``--default-direction``.
    """
    if not frame.subject_class:
        return
    if args.direction != "auto":
        frame.facing, frame.facing_source = args.direction, "override"
    elif math.isfinite(frame.roof_offset) and abs(frame.roof_offset) >= FACING_MIN_OFFSET:
        frame.facing = "left" if frame.roof_offset > 0 else "right"
        frame.facing_source = "roofline"
    elif math.isfinite(frame.end_offset) and abs(frame.end_offset) >= END_MIN_OFFSET:
        frame.facing = "left" if frame.end_offset > 0 else "right"
        frame.facing_source = "taller end"
    elif (math.isfinite(frame.drift_x)
            and frame.drift_response >= MIN_DRIFT_RESPONSE
            and abs(frame.drift_x) >= MIN_DRIFT_PIXELS):
        frame.facing = "right" if frame.drift_x < 0 else "left"
        frame.facing_source = "background drift"
    else:
        frame.facing, frame.facing_source = args.default_direction, "default"


def facing_sign(facing: str) -> int:
    """+1 for a vehicle facing right, -1 for facing left."""
    return 1 if facing == "right" else -1


def find_driver(detections: list[Detection], subject: Detection
                ) -> tuple[tuple[float, float], str]:
    """Driver's head position in analysis pixels, and how it was found."""
    x0, y0, x1, y1 = subject.box
    inside = []
    for det in detections:
        if det.class_id != PERSON_CLASS:
            continue
        px0, py0, px1, py1 = det.box
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y0 + DRIVER_MAX_DEPTH * (y1 - y0):
            inside.append(det)
    if inside:
        det = max(inside, key=lambda d: d.confidence)
        px0, py0, px1, py1 = det.box
        return (float((px0 + px1) / 2), float(py0 + HEAD_DEPTH * (py1 - py0))), "detected"
    return (float((x0 + x1) / 2), float(y0 + COCKPIT_DEPTH * (y1 - y0))), "cockpit estimate"


def measure_horizon(rgb: np.ndarray, mask: np.ndarray | None) -> float:
    """Angle of the dominant near-level background line, in degrees.

    Counterclockwise positive (a line rising to the right is positive).
    NaN when no long line is found.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if mask is not None:
        edges[cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0] = 0
    width = gray.shape[1]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                            minLineLength=int(HORIZON_MIN_LENGTH * width), maxLineGap=10)
    if lines is None:
        return float("nan")
    angles, lengths = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= HORIZON_MAX_ANGLE:
            angles.append(angle)
            lengths.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return float("nan")
    order = np.argsort(angles)
    cumulative = np.cumsum(np.asarray(lengths)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def count_people(detections: list[Detection], subject: Detection | None,
                 frame_area: float) -> int:
    """Count people large enough to distract, excluding occupants of the car."""
    count = 0
    for det in detections:
        if det.class_id != PERSON_CLASS:
            continue
        x0, y0, x1, y1 = det.box
        if (x1 - x0) * (y1 - y0) / frame_area < MIN_PERSON_AREA:
            continue
        if subject is not None:
            sx0, sy0, sx1, sy1 = subject.box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if sx0 <= cx <= sx1 and sy0 <= cy <= sy1:
                continue  # the driver seen through the window
        count += 1
    return count


def drift_thumbnail(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Grayscale thumbnail with the vehicle blanked, for background drift."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    thumb, _ = downscale(gray, DRIFT_LONG_EDGE)
    if mask is not None:
        small_mask, _ = downscale(mask, DRIFT_LONG_EDGE)
        vehicle = small_mask > 0
        if np.any(~vehicle):
            thumb[vehicle] = thumb[~vehicle].mean()
    return thumb


def estimate_drift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Horizontal shift of ``current`` relative to ``previous`` by phase correlation.

    Returns (shift in thumbnail pixels, peak response). A negative shift means
    the background moved left, i.e. the camera panned right.
    """
    if previous.shape != current.shape:
        return float("nan"), 0.0
    height, width = current.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    spectrum_prev = np.fft.fft2((previous - previous.mean()) * window)
    spectrum_curr = np.fft.fft2((current - current.mean()) * window)
    cross = spectrum_curr * np.conj(spectrum_prev)
    cross /= np.abs(cross) + 1e-9
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    response = float(correlation[peak_y, peak_x])
    if peak_x > width // 2:
        peak_x -= width
    return float(peak_x), response


def choose_subject(vehicles: list[Detection]) -> Detection | None:
    """The vehicle to photograph: the largest, unless a car of comparable size is present."""
    largest = max(vehicles, key=lambda d: d.area, default=None)
    if largest is None or largest.class_id == CAR_CLASS:
        return largest
    car = max((d for d in vehicles if d.class_id == CAR_CLASS), key=lambda d: d.area, default=None)
    if car is not None and car.area >= CAR_PREFERENCE_RATIO * largest.area:
        return car
    return largest


def analyze_frame(frame: Frame, rgb: np.ndarray, detector: VehicleDetector) -> np.ndarray:
    """Detect and measure the vehicle in one frame; return its drift thumbnail."""
    frame.height, frame.width = rgb.shape[:2]
    small, factor = downscale(rgb, ANALYSIS_LONG_EDGE)
    frame.analysis_scale = 1.0 / factor
    small_height, small_width = small.shape[:2]
    frame.analysis_size = (small_width, small_height)

    detections = detector(small)
    confident = [d for d in detections if d.confidence >= detector.confidence]
    vehicles = [d for d in confident if d.class_id in VEHICLE_CLASSES]
    subject = choose_subject(vehicles)

    mask = None
    if subject is not None:
        frame.subject_class = VEHICLE_CLASSES[subject.class_id]
        frame.subject_confidence = subject.confidence
        frame.subject_polygon = subject.polygon
        mask = polygon_mask(subject.polygon, small_width, small_height)
        measure_subject(frame, small, mask)
        measure_wheels(frame, small, mask)
        measure_blur(frame, rgb)
        measure_pan_blur(frame, small, mask)
        frame.roof_offset = roof_offset(mask)
        frame.end_offset = end_offset(mask)
        frame.driver_point, frame.driver_source = find_driver(detections, subject)

    frame.horizon = measure_horizon(small, mask)
    frame.people = count_people(confident, subject, float(small_width * small_height))
    return drift_thumbnail(small, mask)


def analyze_all(frames: list[Frame], detector: VehicleDetector,
                args: argparse.Namespace, clip_model: ClipModel | None = None) -> None:
    """Analyze frames in capture order, measuring drift within each pass.

    With ``clip_model``, every frame's vehicle crop is also embedded, for
    splitting passes where the car changes.
    """
    previous_thumb, previous_group, previous_time = None, None, None
    total = len(frames)
    for index, frame in enumerate(frames, 1):
        started = time.perf_counter()
        try:
            rgb, _, _ = load_upright(frame.path)
        except (OSError, ValueError) as exc:
            frame.status, frame.reason = "rejected", f"unreadable: {exc}"
            previous_thumb = None
            print(f"[{index}/{total}] {frame.path.name}: unreadable")
            continue

        thumb = analyze_frame(frame, rgb, detector)
        if clip_model is not None and frame.subject_class:
            small, _ = downscale(rgb, ANALYSIS_LONG_EDGE)
            x0, y0, x1, y1 = vehicle_box(frame)
            frame.car_embedding = clip_model.embed(small[y0:y1, x0:x1])
        if (previous_thumb is not None and previous_group == frame.group
                and frame.capture_time - previous_time <= MAX_DRIFT_INTERVAL):
            frame.drift_x, frame.drift_response = estimate_drift(previous_thumb, thumb)
        previous_thumb, previous_group, previous_time = thumb, frame.group, frame.capture_time
        resolve_facing(frame, args)

        found = frame.subject_class or "no vehicle"
        if frame.facing:
            found += f", facing {frame.facing} ({frame.facing_source})"
        elapsed = time.perf_counter() - started
        print(f"[{index}/{total}] {frame.path.name}  pass {frame.group}  {found}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Grouping, selection, and direction of travel
# ---------------------------------------------------------------------------

def assign_groups(frames: list[Frame], gap_seconds: float) -> None:
    """Split time-ordered frames into passes at gaps longer than ``gap_seconds``."""
    group = 0
    for index, frame in enumerate(frames):
        if index and frame.capture_time - frames[index - 1].capture_time > gap_seconds:
            group += 1
        frame.group = group


def split_passes(frames: list[Frame], gap_seconds: float) -> int:
    """Re-split time-ordered frames into passes using time gaps and car changes.

    A new pass starts after a gap longer than ``gap_seconds``, or after a gap
    longer than SPLIT_MIN_GAP when the car looks different from the last car
    seen. Returns the number of passes.
    """
    group, previous = 0, None
    for index, frame in enumerate(frames):
        if index:
            gap = frame.capture_time - frames[index - 1].capture_time
            changed = (gap > SPLIT_MIN_GAP and previous is not None
                       and frame.car_embedding is not None
                       and float(frame.car_embedding @ previous.car_embedding) < SPLIT_SIMILARITY)
            if gap > gap_seconds or changed:
                group += 1
        frame.group = group
        if frame.car_embedding is not None:
            previous = frame
    return group + 1 if frames else 0


def reject(frame: Frame, reason: str) -> None:
    frame.status, frame.reason = "rejected", reason


def normalize(values: list[float]) -> np.ndarray:
    """Min-max scale to 0..1 within a group.

    A constant feature contributes nothing. Unmeasured values (NaN) are set to
    0.5, the middle of the group's range, so they neither help nor hurt.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    low, high = float(array[finite].min()), float(array[finite].max())
    if high - low < 1e-12:
        scaled = np.zeros_like(array)
    else:
        scaled = (array - low) / (high - low)
    scaled[~finite] = 0.5
    return scaled


FEATURES = {
    "car_sharpness": lambda f: f.car_sharpness,
    "blur": lambda f: f.blur,
    "log_blur_ratio": lambda f: math.log(max(f.blur_ratio, 1e-6)),
    "pan_coherence": lambda f: f.pan_coherence,
    "subject_area": lambda f: f.subject_area,
    "wheel_blur": lambda f: f.wheel_blur,
    "attitude": lambda f: f.attitude,
    "aesthetic": lambda f: f.crop_aesthetic if math.isfinite(f.crop_aesthetic) else f.aesthetic,
    "night": lambda f: f.night,
    "signage": lambda f: f.signage,
    "clip_fraction": lambda f: f.clip_fraction,
    "people": lambda f: float(f.people),
}


def feature_matrix(members: list[Frame]) -> np.ndarray:
    """Normalized feature matrix (frames x FEATURES) for one ranking group."""
    columns = [normalize([extract(f) for f in members]) for extract in FEATURES.values()]
    return np.column_stack(columns)


def weight_vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(name, 0.0) for name in FEATURES])


def reject_in_pass(members: list[Frame], args: argparse.Namespace) -> None:
    """Reject unusable frames of one pass; mark the rest as candidates.

    Frames on the keep list skip every quality test; they only need a vehicle.
    """
    candidates = []
    for frame in members:
        if frame.status == "rejected":
            continue
        if not frame.subject_class:
            reject(frame, "no vehicle detected")
        elif frame.keep:
            frame.status, frame.reason = "candidate", "on keep list"
        elif frame.touches_edge:
            reject(frame, "vehicle cut off by frame edge")
        elif hard_blur_reason(frame):
            reject(frame, hard_blur_reason(frame))
        elif frame.car_sharpness < args.sharpness_floor:
            reject(frame, "vehicle below absolute sharpness floor")
        else:
            candidates.append(frame)
    if not candidates:
        return

    sharpest = max(frame.car_sharpness for frame in candidates)
    for frame in candidates:
        if frame.keep:
            continue
        relative = frame.car_sharpness / sharpest if sharpest > 0 else 1.0
        if relative < args.relative_sharpness:
            reject(frame, f"vehicle soft ({relative:.0%} of sharpest in pass)")
        else:
            frame.status = "candidate"


def drop_weak_frames(members: list[Frame], args: argparse.Namespace, label: str) -> None:
    """Reject blurry frames when the group has a better one; always keep one.

    A frame is weak when ``weak_reasons`` finds a problem. Weak frames are
    rejected if the group (driver) has any frame without problems; otherwise
    the weak frame with the lowest display blur is kept as the best available.
    """
    candidates = [f for f in members if f.status == "candidate"]
    weak = {id(f): weak_reasons(f, args) for f in candidates if not f.keep}
    weak = {key: reasons for key, reasons in weak.items() if reasons}
    good = [f for f in candidates if id(f) not in weak]
    for frame in good:
        frame.quality = "good"
    best = None
    if not good and candidates:
        best = min(candidates, key=lambda f: (f.display_blur
                                              if math.isfinite(f.display_blur) else math.inf))
        best.quality = "best available"
    for frame in candidates:
        if id(frame) in weak and frame is not best:
            reject(frame, f"{'; '.join(weak[id(frame)])}; better photos of {label} exist"
                          if good else f"{'; '.join(weak[id(frame)])}")


def view_of(frame: Frame) -> str:
    """Facing plus side-on or angled, e.g. "right side" or "left angled"."""
    xs, ys = frame.subject_polygon[:, 0], frame.subject_polygon[:, 1]
    aspect = float(xs.max() - xs.min()) / max(1.0, float(ys.max() - ys.min()))
    return f"{frame.facing} {'side' if aspect >= SIDE_VIEW_ASPECT else 'angled'}"


def drop_soft_duplicates(members: list[Frame]) -> None:
    """Reject soft frames whose view the same group already has in a crisp frame."""
    candidates = [f for f in members if f.status == "candidate"]
    crisp = sorted((f for f in candidates if math.isfinite(f.blur) and f.blur <= CRISP_BLUR),
                   key=lambda f: f.blur)
    for frame in candidates:
        if frame.keep or not math.isfinite(frame.blur) or frame.blur <= CRISP_BLUR:
            continue
        view = view_of(frame)
        twin = next((c for c in crisp if view_of(c) == view), None)
        if twin is not None:
            reject(frame, f"soft duplicate (blur {frame.blur:.2f}; "
                          f"{twin.path.name} is a crisp {view} shot)")


def read_keep_list(folder: Path) -> set[str]:
    """Lower-case file stems listed in the shoot's keep list, if it exists."""
    path = folder / KEEP_LIST_NAME
    if not path.is_file():
        return set()
    stems = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            stems.add(Path(entry).stem.lower())
    return stems


def rank_group(members: list[Frame], weights: dict[str, float], label: str,
               per_driver: int = PHOTOS_PER_DRIVER,
               duplicate_similarity: float = DUPLICATE_SIMILARITY) -> None:
    """Rank one group's candidates and choose which to export.

    In rank order, a photo is chosen unless the group already has
    ``per_driver`` photos or its crop looks near-identical to one already
    chosen; keep-list photos are always chosen. The first chosen is the hero.
    Photos not chosen are marked "passed" and not exported.
    """
    candidates = [f for f in members if f.status == "candidate"]
    if not candidates:
        return
    scores = feature_matrix(candidates) @ weight_vector(weights)
    chosen: list[Frame] = []
    for rank, index in enumerate(np.argsort(-scores, kind="stable"), 1):
        frame = candidates[index]
        frame.rank, frame.score = rank, float(scores[index])
        twin, similarity = None, -1.0
        if frame.crop_embedding is not None:
            for other in chosen:
                if other.crop_embedding is not None:
                    value = float(frame.crop_embedding @ other.crop_embedding)
                    if value > similarity:
                        twin, similarity = other, value
        duplicate = twin is not None and similarity >= duplicate_similarity
        if frame.keep or (len(chosen) < per_driver and not duplicate):
            chosen.append(frame)
            frame.status = "hero" if len(chosen) == 1 else "keeper"
            frame.reason = f"best of {label}" if rank == 1 else f"rank {rank} of {label}"
        elif duplicate:
            frame.status = "passed"
            frame.reason = f"near-identical to {twin.path.name} ({similarity:.2f})"
        else:
            frame.status = "passed"
            frame.reason = f"rank {rank} of {label}; {label} already has {per_driver} photos"
    assert sum(f.status == "hero" for f in members) == 1, f"{label}: expected one hero"


def ranking_groups(frames: list[Frame], select_by: str) -> dict[str, list[Frame]]:
    """Group frames by driver or by pass for ranking."""
    groups: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        key = frame.driver if select_by == "driver" and frame.driver else f"pass {frame.group}"
        groups[key].append(frame)
    return groups


# ---------------------------------------------------------------------------
# Aesthetics, car numbers, and appearance
# ---------------------------------------------------------------------------

def torch_device(requested: str | None) -> str:
    """Resolve a torch device string, accepting Ultralytics-style GPU indices."""
    import torch

    if requested:
        return f"cuda:{requested}" if requested.isdigit() else requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipModel:
    """CLIP ViT-L/14 image encoder with the LAION aesthetic predictor head.

    One encoder serves both purposes: whole-frame embeddings feed the
    aesthetic head, and vehicle-crop embeddings drive appearance matching.
    """

    def __init__(self, device: str) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        # Half precision on the GPU halves memory, leaving room for the
        # number-reading model served by Ollama.
        half = device.startswith("cuda")
        self.dtype = torch.float16 if half else torch.float32
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_ARCHITECTURE, pretrained=CLIP_PRETRAINED, device=device,
            precision="fp16" if half else "fp32")
        self.model.eval()
        self.head = self._load_aesthetic_head()

    def _load_aesthetic_head(self):
        torch, nn = self.torch, self.torch.nn
        path = CACHE_DIR / "laion_aesthetic_vit_l14.pth"
        if not path.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print("Downloading aesthetic predictor weights ...")
            torch.hub.download_url_to_file(AESTHETIC_HEAD_URL, str(path))
        # Layer layout of the published predictor; dropout layers carry no weights.
        head = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        state = torch.load(path, map_location="cpu")
        prefix = "layers."
        head.load_state_dict({(k[len(prefix):] if k.startswith(prefix) else k): v
                              for k, v in state.items()})
        return head.to(self.device).eval()

    def _encode(self, rgb: np.ndarray):
        image = self.preprocess(Image.fromarray(np.ascontiguousarray(rgb)))
        with self.torch.no_grad():
            features = self.model.encode_image(
                image.unsqueeze(0).to(self.device, dtype=self.dtype)).float()
        return features / features.norm(dim=-1, keepdim=True)

    def aesthetic(self, rgb: np.ndarray) -> float:
        """Predicted aesthetic rating, roughly on a 1 to 10 scale."""
        with self.torch.no_grad():
            return float(self.head(self._encode(rgb)).item())

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """Unit-length appearance embedding."""
        return self._encode(rgb).cpu().numpy()[0]


class NumberReader:
    """Reads the car number from a vehicle crop with EasyOCR."""

    def __init__(self, gpu: bool) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, rgb: np.ndarray) -> tuple[str, float]:
        """Return (number, score); score weights confidence by text height."""
        crop_height = rgb.shape[0]
        best_number, best_score = "", 0.0
        for box, text, confidence in self.reader.readtext(rgb, allowlist=NUMBER_ALLOWLIST):
            match = NUMBER_PATTERN.fullmatch(text.replace(" ", "").upper())
            if not match or confidence < NUMBER_MIN_CONFIDENCE:
                continue
            digits = match.group(1)
            ys = [point[1] for point in box]
            text_height = max(ys) - min(ys)
            if text_height < NUMBER_MIN_HEIGHT * crop_height:
                continue  # sponsor decals, phone numbers, and other small text
            score = float(confidence) * text_height / crop_height
            if score > best_score:
                best_number, best_score = digits, score
        return best_number, best_score


def vehicle_box(frame: Frame, padding: float = 0.04) -> tuple[int, int, int, int]:
    """Padded vehicle bounding box (x0, y0, x1, y1) in analysis pixels."""
    width, height = frame.analysis_size
    xs, ys = frame.subject_polygon[:, 0], frame.subject_polygon[:, 1]
    pad_x = padding * float(xs.max() - xs.min())
    pad_y = padding * float(ys.max() - ys.min())
    return (max(0, int(xs.min() - pad_x)), max(0, int(ys.min() - pad_y)),
            min(width, int(math.ceil(xs.max() + pad_x))),
            min(height, int(math.ceil(ys.max() + pad_y))))


class VisionNumberReader:
    """Reads the car number by asking a vision-language model served by Ollama.

    Unlike OCR, the model understands the number panel: it separates the
    number from class letters and ignores plates and sponsor text. If the
    server stops answering partway through a run, the remaining frames are
    read with EasyOCR instead of stopping the run.
    """

    def __init__(self, url: str, model: str, gpu: bool) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.gpu = gpu
        self.fallback: NumberReader | None = None
        self.failed = False
        info = self._post("/api/show", {"model": model}, OLLAMA_CONNECT_TIMEOUT)
        if "vision" not in (info.get("capabilities") or []):
            raise RuntimeError(f"{model} cannot read images")

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(self.url + path, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def read(self, rgb: np.ndarray) -> tuple[str, float]:
        """Return (number, score); score is the model's confidence as a weight."""
        if not self.failed:
            error: OSError | None = None
            for wait in (0, *OLLAMA_RETRY_WAITS):
                if error is not None:
                    print(f"\n{self.model} read failed ({error}); retrying in {wait}s ...")
                    time.sleep(wait)
                try:
                    return self._ask(rgb)
                except OSError as exc:  # refused, reset, timed out, or a server error
                    error = exc
            self.failed = True
            print(f"\n{self.model} at {self.url} stopped answering ({error}); "
                  "reading the remaining numbers with EasyOCR.")
            try:
                self.fallback = NumberReader(gpu=self.gpu)
            except Exception as fallback_exc:  # not installed or failed to load
                print(f"EasyOCR unavailable ({fallback_exc}); remaining numbers not read.")
        return self.fallback.read(rgb) if self.fallback is not None else ("", 0.0)

    def _ask(self, rgb: np.ndarray) -> tuple[str, float]:
        image = Image.fromarray(np.ascontiguousarray(rgb))
        image.thumbnail((VLM_IMAGE_EDGE, VLM_IMAGE_EDGE))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        reply = self._post("/api/generate", {
            "model": self.model, "prompt": VLM_NUMBER_PROMPT, "format": "json",
            "stream": False, "think": False, "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0},
            "images": [base64.b64encode(buffer.getvalue()).decode()],
        }, OLLAMA_READ_TIMEOUT)
        try:
            answer = json.loads(reply.get("response", ""))
        except json.JSONDecodeError:
            return "", 0.0
        digits = "".join(ch for ch in str(answer.get("number") or "") if ch.isdigit())
        score = VLM_CONFIDENCE_SCORES.get(str(answer.get("confidence")).lower(), 0.0)
        if not 1 <= len(digits) <= NUMBER_MAX_DIGITS or score == 0.0:
            return "", 0.0
        return digits, score


def load_number_reader(args: argparse.Namespace, device: str
                       ) -> NumberReader | VisionNumberReader | None:
    """The configured number reader, falling back from the vision model to EasyOCR."""
    if args.number_reader == "vlm":
        try:
            print(f"Connecting to {args.ollama_model} at {args.ollama_url} ...")
            return VisionNumberReader(args.ollama_url, args.ollama_model,
                                      gpu=device.startswith("cuda"))
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
            print(f"Vision number reader unavailable ({exc}); falling back to EasyOCR.")
    try:
        print("Loading EasyOCR number reader ...")
        return NumberReader(gpu=device.startswith("cuda"))
    except ImportError:
        print("easyocr not installed; number reading skipped.")
    except Exception as exc:  # download or model-construction failure
        print(f"Number reader unavailable ({exc}); number reading skipped.")
    return None


def load_clip(args: argparse.Namespace) -> ClipModel | None:
    """The CLIP model, or None when disabled or unavailable."""
    if args.no_clip:
        return None
    try:
        device = torch_device(args.device)
        print(f"Loading CLIP {CLIP_ARCHITECTURE} on {device} ...")
        return ClipModel(device)
    except ImportError:
        print("PyTorch or open_clip_torch not installed; aesthetic scoring, appearance "
              "matching, and pass splitting by car skipped.")
    except Exception as exc:  # download or model-construction failure
        print(f"CLIP unavailable ({exc}); aesthetic scoring, appearance matching, "
              "and pass splitting by car skipped.")
    return None


def load_reader(args: argparse.Namespace) -> NumberReader | VisionNumberReader | None:
    """The number reader, or None when disabled or unavailable."""
    if args.no_ocr:
        return None
    try:
        device = torch_device(args.device)
    except ImportError:
        device = "cpu"
    return load_number_reader(args, device)


def enrich_candidates(frames: list[Frame], args: argparse.Namespace,
                      clip_model: ClipModel | None,
                      number_reader: NumberReader | VisionNumberReader | None) -> None:
    """Score aesthetics, read numbers, and embed appearance for candidates only.

    These models are the slow part of the pipeline, so they run only on frames
    that survived rejection. Appearance embeddings are computed for the
    ``--embed-top`` sharpest candidates of each pass, which is enough to
    characterize the car.
    """
    candidates = [f for f in frames if f.status == "candidate"]
    if not candidates or (clip_model is None and number_reader is None):
        return

    by_pass: dict[int, list[Frame]] = defaultdict(list)
    for frame in candidates:
        by_pass[frame.group].append(frame)
    embed_ids = {id(f) for members in by_pass.values()
                 for f in sorted(members, key=lambda f: -f.car_sharpness)[:args.embed_top]}

    for index, frame in enumerate(candidates, 1):
        started = time.perf_counter()
        rgb, _, _ = load_upright(frame.path)
        small, _ = downscale(rgb, ANALYSIS_LONG_EDGE)
        x0, y0, x1, y1 = vehicle_box(frame)

        if clip_model is not None:
            frame.aesthetic = clip_model.aesthetic(small)
            if id(frame) in embed_ids:
                frame.embedding = clip_model.embed(small[y0:y1, x0:x1])

        if number_reader is not None:
            scale = frame.analysis_scale
            crop = rgb[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)]
            crop, _ = downscale(crop, NUMBER_CROP_LONG_EDGE)
            frame.car_number, frame.number_score = number_reader.read(crop)

        elapsed = time.perf_counter() - started
        details = []
        if math.isfinite(frame.aesthetic):
            details.append(f"aesthetic {frame.aesthetic:.2f}")
        if frame.car_number:
            details.append(f"number {frame.car_number}")
        print(f"[{index}/{len(candidates)}] {frame.path.name}  "
              f"{'  '.join(details) or '-'}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Driver identification
# ---------------------------------------------------------------------------

def pass_number(members: list[Frame]) -> str:
    """The pass's car number by confidence-weighted vote, or "" if unclear."""
    totals: dict[str, float] = defaultdict(float)
    for frame in members:
        if frame.car_number:
            totals[frame.car_number] += frame.number_score
    if not totals:
        return ""
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    number, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return number if score >= NUMBER_MIN_SCORE and score >= NUMBER_MARGIN * runner_up else ""


def pass_embedding(members: list[Frame]) -> np.ndarray | None:
    """Mean unit-length appearance embedding of a pass, if any were computed."""
    vectors = [f.embedding for f in members if f.embedding is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else None


def identify_drivers(passes: dict[int, list[Frame]], threshold: float
                     ) -> tuple[dict[int, str], dict[int, str], dict[int, float]]:
    """Merge passes into drivers.

    Returns (driver label per pass, number per pass, appearance similarity
    that joined each unnumbered pass). Passes reading the same number merge;
    an unnumbered pass joins its most similar pass at or above ``threshold``;
    a merge that would put two different numbers in one driver is refused.
    """
    group_ids = sorted(passes)
    numbers = {g: pass_number(passes[g]) for g in group_ids}
    embeddings = {g: pass_embedding(passes[g]) for g in group_ids}
    parent = {g: g for g in group_ids}
    cluster_numbers = {g: ({numbers[g]} if numbers[g] else set()) for g in group_ids}

    def find(g: int) -> int:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return True
        merged = cluster_numbers[root_a] | cluster_numbers[root_b]
        if len(merged) > 1:
            return False
        parent[root_b] = root_a
        cluster_numbers[root_a] = merged
        return True

    first_pass_with_number: dict[str, int] = {}
    for g in group_ids:
        number = numbers[g]
        if number:
            if number in first_pass_with_number:
                union(first_pass_with_number[number], g)
            else:
                first_pass_with_number[number] = g

    match_similarity = {g: float("nan") for g in group_ids}
    for g in group_ids:
        if numbers[g] or embeddings[g] is None:
            continue
        similarities = sorted(((float(embeddings[g] @ embeddings[other]), other)
                               for other in group_ids
                               if other != g and embeddings[other] is not None),
                              reverse=True)
        for similarity, other in similarities:
            if similarity < threshold:
                break
            if union(g, other):
                match_similarity[g] = similarity
                break

    clusters: dict[int, list[int]] = defaultdict(list)
    for g in group_ids:
        clusters[find(g)].append(g)
    labels: dict[int, str] = {}
    unnumbered = 0
    for root in sorted(clusters, key=lambda r: min(clusters[r])):
        if cluster_numbers[root]:
            label = f"#{next(iter(cluster_numbers[root]))}"
        else:
            unnumbered += 1
            label = f"D{unnumbered:02d}"
        for g in clusters[root]:
            labels[g] = label
    return labels, numbers, match_similarity


# ---------------------------------------------------------------------------
# Learning ranking weights from hand-picked frames
# ---------------------------------------------------------------------------

def mark_picks(frames: list[Frame], picks_folder: Path) -> tuple[int, list[str]]:
    """Flag frames that match photos in ``picks_folder``.

    A pick matches the original with the same EXIF capture time when exactly
    one original has it; otherwise the original whose file name is the
    longest prefix of the pick's name (e.g. IMG_1234 for IMG_1234-Edit).
    """
    picks = sorted(p for p in picks_folder.iterdir()
                   if p.is_file() and p.suffix.lower() in JPEG_EXTENSIONS)
    by_time: dict[float, list[Frame]] = defaultdict(list)
    for frame in frames:
        if frame.time_source == "exif":
            by_time[round(frame.capture_time, 3)].append(frame)
    longest_first = sorted(frames, key=lambda f: -len(f.path.stem))

    matched, unmatched = 0, []
    for pick in picks:
        capture_time, source = read_capture_time(pick)
        target = None
        if source == "exif":
            same_time = by_time.get(round(capture_time, 3), [])
            if len(same_time) == 1:
                target = same_time[0]
        if target is None:
            stem = pick.stem.lower()
            target = next((f for f in longest_first if stem.startswith(f.path.stem.lower())), None)
        if target is None:
            unmatched.append(pick.name)
        else:
            target.picked = True
            matched += 1
    return matched, unmatched


def training_data(frames: list[Frame], select_by: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(normalized features, picked flags) for every group with a pick and a non-pick."""
    data = []
    groups = ranking_groups(frames, select_by)
    for key in sorted(groups):
        candidates = [f for f in groups[key] if f.status == "candidate"]
        labels = np.array([f.picked for f in candidates], dtype=bool)
        if len(candidates) >= 2 and labels.any() and (~labels).any():
            data.append((feature_matrix(candidates), labels))
    return data


def fit_pairwise(data: list[tuple[np.ndarray, np.ndarray]], l2: float = 0.01,
                 iterations: int = 3000, rate: float = 0.5) -> np.ndarray | None:
    """Fit weights so each pick outscores the other frames of its group.

    Logistic regression on feature differences (pick minus non-pick) without
    an intercept, i.e. a Bradley-Terry preference model. Weights are scaled
    so their absolute values sum to 1.
    """
    differences = [features[i] - features[j]
                   for features, labels in data
                   for i in np.flatnonzero(labels)
                   for j in np.flatnonzero(~labels)]
    if not differences:
        return None
    design = np.asarray(differences)
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = -(design.T @ (1.0 - probability)) / len(design) + l2 * weights
        weights -= rate * gradient
    total = float(np.abs(weights).sum())
    return weights / total if total > 0 else weights


def agreement(data: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray) -> tuple[float, float]:
    """Fraction of groups whose top-scored frame, and top three, include a pick."""
    if not data:
        return float("nan"), float("nan")
    top1 = top3 = 0
    for features, labels in data:
        order = np.argsort(-(features @ weights), kind="stable")
        top1 += bool(labels[order[0]])
        top3 += bool(labels[order[:3]].any())
    return top1 / len(data), top3 / len(data)


def learn_weights(frames: list[Frame], weights: dict[str, float],
                  args: argparse.Namespace, output_dir: Path) -> None:
    """Fit ranking weights to hand-picked frames, report agreement, and save them."""
    matched, unmatched = mark_picks(frames, args.learn_from)
    print(f"\nPicks matched to originals: {matched}"
          + (f"; unmatched: {len(unmatched)}" if unmatched else ""))
    for name in unmatched[:10]:
        print(f"  unmatched pick: {name}")

    rejected_picks = Counter(f.reason.split(" (")[0] for f in frames
                             if f.picked and f.status == "rejected")
    for reason, count in rejected_picks.most_common():
        print(f"  pick rejected before ranking, {reason}: {count}")

    data = training_data(frames, args.select_by)
    if not data:
        print("No group contains both a pick and an unpicked candidate; nothing to learn.")
        return

    current = weight_vector(weights)
    learned = fit_pairwise(data)
    print(f"\nGroups used for learning: {len(data)}")
    print(f"{'':24}{'top-1':>8}{'top-3':>8}")
    for label, vector in (("current weights", current), ("learned, same data", learned)):
        top1, top3 = agreement(data, vector)
        print(f"{label:24}{top1:>8.0%}{top3:>8.0%}")

    # Two-fold cross-validation: fit on half the groups, score on the other half.
    if len(data) >= 4:
        halves = (data[0::2], data[1::2])
        hits1 = hits3 = 0.0
        for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            fold_weights = fit_pairwise(train)
            if fold_weights is None:
                continue
            top1, top3 = agreement(test, fold_weights)
            hits1 += top1 * len(test)
            hits3 += top3 * len(test)
        print(f"{'learned, cross-checked':24}{hits1 / len(data):>8.0%}{hits3 / len(data):>8.0%}")
    else:
        print("Fewer than 4 groups: cross-checked agreement not computed.")

    learned_weights = {name: round(float(w), 4) for name, w in zip(FEATURES, learned)}
    print("\nLearned weights:")
    for name, value in sorted(learned_weights.items(), key=lambda item: -abs(item[1])):
        print(f"  {name:16}{value:+.3f}")
    path = output_dir / "learned_weights.json"
    path.write_text(json.dumps(learned_weights, indent=2), encoding="utf-8")
    print(f"Saved {path}; apply with --weights \"{path}\"")


def load_weights(path: Path | None) -> dict[str, float]:
    """Default ranking weights, overridden by a JSON file when given."""
    weights = dict(RANKING_WEIGHTS)
    if path is None:
        return weights
    loaded = json.loads(path.read_text(encoding="utf-8"))
    for name, value in loaded.items():
        if name in FEATURES:
            weights[name] = float(value)
        else:
            print(f"Ignoring unknown weight '{name}' in {path.name}")
    return weights


# ---------------------------------------------------------------------------
# Tilt and crop
# ---------------------------------------------------------------------------

def rotation_matrix(width: int, height: int, angle: float) -> tuple[np.ndarray, int, int]:
    """Rotation about the image center onto a canvas large enough to hold it.

    Positive angles rotate counterclockwise, raising the right side.
    """
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    canvas_width = int(math.ceil(height * sin + width * cos))
    canvas_height = int(math.ceil(height * cos + width * sin))
    matrix[0, 2] += canvas_width / 2.0 - center[0]
    matrix[1, 2] += canvas_height / 2.0 - center[1]
    return matrix, canvas_width, canvas_height


def plan_crop(mask: np.ndarray, angle: float, anchor_x: float, anchor_y: float,
              subject_width: float, aspect: float) -> CropPlan | None:
    """Find the crop that places the vehicle closest to the composition anchor.

    The crop keeps ``aspect``, contains the whole vehicle, contains no blank
    corners from the rotation, and is as large as possible while the vehicle
    centroid lands within ``ANCHOR_TOLERANCE`` of the anchor point.
    """
    height, width = mask.shape
    matrix, canvas_width, canvas_height = rotation_matrix(width, height, angle)

    valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), matrix,
                           (canvas_width, canvas_height), flags=cv2.INTER_NEAREST,
                           borderValue=0)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))  # margin against edge fringe
    subject = cv2.warpAffine(mask, matrix, (canvas_width, canvas_height),
                             flags=cv2.INTER_NEAREST, borderValue=0)

    ys, xs = np.nonzero(subject)
    if xs.size == 0:
        return None
    box_x0, box_x1 = int(xs.min()), int(xs.max()) + 1
    box_y0, box_y1 = int(ys.min()), int(ys.max()) + 1
    centroid_x, centroid_y = float(xs.mean()), float(ys.mean())
    box_width, box_height = box_x1 - box_x0, box_y1 - box_y0
    pad_x, pad_y = SUBJECT_PADDING * box_width, SUBJECT_PADDING * box_height

    min_width = max(box_width + 2 * pad_x, (box_height + 2 * pad_y) * aspect)
    max_width = min(canvas_width, canvas_height * aspect)
    if min_width > max_width:
        return None
    target_width = min(max(min_width, box_width / subject_width), max_width)

    # Summed-area table of invalid pixels: a window is usable when its sum is 0.
    integral = cv2.integral((valid == 0).astype(np.uint8))

    best: tuple[float, CropPlan] | None = None
    for crop_width_f in np.linspace(target_width, min_width, CROP_SEARCH_STEPS):
        crop_width = int(round(crop_width_f))
        crop_height = int(round(crop_width_f / aspect))
        if not (0 < crop_width <= canvas_width and 0 < crop_height <= canvas_height):
            continue

        rows = canvas_height + 1 - crop_height
        cols = canvas_width + 1 - crop_width
        window_sums = (integral[crop_height:, crop_width:]
                       - integral[:rows, crop_width:]
                       - integral[crop_height:, :cols]
                       + integral[:rows, :cols])

        x_low = max(0, math.ceil(box_x1 + pad_x - crop_width))
        x_high = min(canvas_width - crop_width, math.floor(box_x0 - pad_x))
        y_low = max(0, math.ceil(box_y1 + pad_y - crop_height))
        y_high = min(canvas_height - crop_height, math.floor(box_y0 - pad_y))
        if x_low > x_high or y_low > y_high:
            continue

        free_y, free_x = np.nonzero(window_sums[y_low:y_high + 1, x_low:x_high + 1] == 0)
        if free_x.size == 0:
            continue
        ideal_x = centroid_x - anchor_x * crop_width
        ideal_y = centroid_y - anchor_y * crop_height
        errors = np.hypot((free_x + x_low - ideal_x) / crop_width,
                          (free_y + y_low - ideal_y) / crop_height)
        pick = int(np.argmin(errors))
        plan = CropPlan(angle, int(free_x[pick] + x_low), int(free_y[pick] + y_low),
                        crop_width, crop_height, canvas_width, canvas_height)
        error = float(errors[pick])
        if error <= ANCHOR_TOLERANCE:
            return plan
        if best is None or error < best[0]:
            best = (error, plan)

    return best[1] if best else None


def render_crop(rgb: np.ndarray, plan: CropPlan) -> np.ndarray:
    """Apply a crop plan made at analysis resolution to the full-size image."""
    height, width = rgb.shape[:2]
    matrix, canvas_width, canvas_height = rotation_matrix(width, height, plan.angle)
    if plan.angle:
        rotated = cv2.warpAffine(rgb, matrix, (canvas_width, canvas_height),
                                 flags=cv2.INTER_LANCZOS4, borderValue=0)
    else:
        rotated = rgb

    scale_x = canvas_width / plan.canvas_width
    scale_y = canvas_height / plan.canvas_height
    crop_width = min(canvas_width, int(round(plan.width * scale_x)))
    crop_height = min(canvas_height, int(round(plan.height * scale_y)))
    x0 = min(max(0, int(round(plan.x * scale_x))), canvas_width - crop_width)
    y0 = min(max(0, int(round(plan.y * scale_y))), canvas_height - crop_height)
    return rotated[y0:y0 + crop_height, x0:x0 + crop_width]


@dataclass
class Composition:
    """The chosen tilt and crop for one photo, and how they were chosen.

    Points are in rotated analysis-canvas pixels; ``rotation`` is in degrees,
    counterclockwise positive.
    """

    plan: CropPlan | None
    rule: str
    rotation: float
    driver_frac: tuple[float, float]
    nose: tuple[float, float]
    tail: tuple[float, float]
    nose_up_reachable: bool
    score: float = float("nan")
    fill: float = float("nan")                  # vehicle width / crop width
    crop_aesthetic: float = float("nan")
    night: float = float("nan")
    signage: float = float("nan")
    candidates: list[dict] = field(default_factory=list)
    alternatives: list["Composition"] = field(default_factory=list)   # looser fills


def transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
            float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]))


def nose_and_tail(mask: np.ndarray, facing: str
                  ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centroids of the front and rear slices of the vehicle."""
    ys, xs = np.nonzero(mask)
    x0, x1 = float(xs.min()), float(xs.max())
    band = max(1.0, NOSE_SLICE * (x1 - x0))
    right, left = xs >= x1 - band, xs <= x0 + band
    right_point = (float(xs[right].mean()), float(ys[right].mean()))
    left_point = (float(xs[left].mean()), float(ys[left].mean()))
    return (right_point, left_point) if facing == "right" else (left_point, right_point)


def nose_rise(nose: tuple[float, float], tail: tuple[float, float]) -> float:
    """Degrees the nose sits above the tail along the car; negative is nose-down."""
    return math.degrees(math.atan2(tail[1] - nose[1], abs(nose[0] - tail[0]) + 1e-9))


def tilt_magnitudes(tilt: float, rise: float) -> list[float]:
    """Candidate tilt magnitudes around ``tilt``, each raised enough to put the nose up."""
    if tilt <= 0:
        return [0.0]
    required = MIN_NOSE_RISE - rise
    magnitudes: list[float] = []
    for factor in TILT_FACTORS:
        magnitude = round(min(MAX_TILT, max(tilt * factor, required)), 1)
        if magnitude not in magnitudes:
            magnitudes.append(magnitude)
    return magnitudes


def compose(mask: np.ndarray, driver: tuple[float, float], facing: str, tilt: float,
            horizon: float, aspect: float, max_fill: float = DEFAULT_MAX_FILL) -> Composition:
    """Choose the tilt and a tight crop, steered toward the rule of thirds.

    Every candidate is evaluated in the rotated frame, so the composition is
    measured on what is saved. Facing right puts the tail close to the left
    edge and the open space ahead of the nose on the right, with the driver
    as near the left vertical third line as the tight crop allows; facing
    left mirrors it. The tightest fill level (up to ``max_fill``) with any
    layout that fits wins; the best layout of each looser level is returned
    in ``alternatives``. Rotation is counterclockwise for a right-facing car
    and clockwise for a left-facing one, which raises the nose in both cases.
    """
    sign = facing_sign(facing)
    height, width = mask.shape
    nose, tail = nose_and_tail(mask, facing)
    rise = nose_rise(nose, tail)
    reachable = tilt <= 0 or rise + MAX_TILT >= MIN_NOSE_RISE
    horizon_rise = sign * horizon      # background tilt in the direction of travel
    column = 1 / 3 if sign > 0 else 2 / 3
    weights = COMPOSE_WEIGHTS

    candidates: list[dict] = []
    level_best: dict[float, Composition] = {}
    for magnitude in tilt_magnitudes(tilt, rise):
        angle = sign * magnitude
        matrix, canvas_width, canvas_height = rotation_matrix(width, height, angle)
        size = (canvas_width, canvas_height)
        valid = cv2.warpAffine(np.full_like(mask, 255), matrix, size,
                               flags=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))  # margin against edge fringe
        integral = cv2.integral((valid == 0).astype(np.uint8))
        subject = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST, borderValue=0)
        ys, xs = np.nonzero(subject)
        if xs.size == 0:
            continue
        box_x0, box_x1 = int(xs.min()), int(xs.max()) + 1
        box_y0, box_y1 = int(ys.min()), int(ys.max()) + 1
        driver_x, driver_y = transform_point(matrix, driver)
        rotated_nose, rotated_tail = transform_point(matrix, nose), transform_point(matrix, tail)
        if math.isfinite(horizon_rise):
            horizon_term = 1.0 - min(1.0, abs(horizon_rise + magnitude - tilt) / max(tilt, 1.0))
        else:
            horizon_term = 0.5

        for row in DRIVER_ROWS:
            for fill in (f for f in FILL_LEVELS if f <= max_fill + 1e-9):
                crop_width = int(round((box_x1 - box_x0) / fill))
                crop_height = int(round(crop_width / aspect))
                if not (0 < crop_width <= canvas_width and 0 < crop_height <= canvas_height):
                    continue
                ideal_x = driver_x - column * crop_width
                ideal_y = driver_y - row * crop_height
                tail_room, lead_room = TAIL_MARGIN * crop_width, LEAD_MIN * crop_width
                vertical_room = VERTICAL_MARGIN * crop_height
                if sign > 0:   # tail at the left edge, lead space on the right
                    x_min, x_max = box_x1 + lead_room - crop_width, box_x0 - tail_room
                else:          # tail at the right edge, lead space on the left
                    x_min, x_max = box_x1 + tail_room - crop_width, box_x0 - lead_room

                x_low = max(0, math.ceil(x_min))
                x_high = min(canvas_width - crop_width, math.floor(x_max))
                y_low = max(0, math.ceil(box_y1 + vertical_room - crop_height))
                y_high = min(canvas_height - crop_height, math.floor(box_y0 - vertical_room))
                if x_low > x_high or y_low > y_high:
                    continue

                cols = np.arange(x_low, x_high + 1)
                rows = np.arange(y_low, y_high + 1)
                blank = (integral[np.ix_(rows + crop_height, cols + crop_width)]
                         - integral[np.ix_(rows, cols + crop_width)]
                         - integral[np.ix_(rows + crop_height, cols)]
                         + integral[np.ix_(rows, cols)])
                free_y, free_x = np.nonzero(blank == 0)
                if free_x.size == 0:
                    continue
                errors = np.hypot((cols[free_x] - ideal_x) / crop_width,
                                  (rows[free_y] - ideal_y) / crop_height)
                pick = int(np.argmin(errors))
                x, y = int(cols[free_x[pick]]), int(rows[free_y[pick]])

                lead_gap = (x + crop_width - box_x1) if sign > 0 else (box_x0 - x)
                terms = {
                    "thirds": max(0.0, 1.0 - float(errors[pick]) / THIRDS_TOLERANCE),
                    "lead": min(1.0, lead_gap / (crop_width / 3)),
                    "horizon": horizon_term,
                }
                score = sum(weights[name] * value for name, value in terms.items())
                driver_frac = ((driver_x - x) / crop_width, (driver_y - y) / crop_height)
                candidates.append({
                    "rotation": round(angle, 1), "row": "upper" if row < 0.5 else "lower",
                    "fill": fill, "score": round(score, 3),
                    "driver": [round(driver_frac[0], 3), round(driver_frac[1], 3)],
                    "terms": {name: round(value, 3) for name, value in terms.items()},
                })
                if fill not in level_best or score > level_best[fill].score:
                    plan = CropPlan(angle, x, y, crop_width, crop_height,
                                    canvas_width, canvas_height)
                    level_best[fill] = Composition(
                        plan, "thirds", angle, driver_frac, rotated_nose, rotated_tail,
                        reachable, score, fill)

    if level_best:
        tight, *looser = [level_best[fill] for fill in sorted(level_best, reverse=True)]
        tight.candidates, tight.alternatives = candidates, looser
        return tight

    # No thirds layout fits inside the image: center the car, keeping nose-up
    # tilt if any tilted crop fits.
    magnitudes = sorted(tilt_magnitudes(tilt, rise), key=lambda m: abs(m - tilt))
    for magnitude in magnitudes + ([0.0] if 0.0 not in magnitudes else []):
        plan = plan_crop(mask, sign * magnitude, 0.5, 0.5, CENTER_SUBJECT_WIDTH, aspect)
        if plan is None:
            continue
        matrix, _, _ = rotation_matrix(width, height, plan.angle)
        driver_x, driver_y = transform_point(matrix, driver)
        rule = ("centered: no thirds layout fits" if magnitude or tilt <= 0
                else "centered, untilted: no tilted crop fits")
        return Composition(plan, rule, plan.angle,
                           ((driver_x - plan.x) / plan.width, (driver_y - plan.y) / plan.height),
                           transform_point(matrix, nose), transform_point(matrix, tail),
                           reachable, candidates=candidates)

    return Composition(None, "uncropped: no crop fits", 0.0,
                       (driver[0] / width, driver[1] / height), nose, tail, reachable,
                       candidates=candidates)


def is_signage_word(word: str, target: str) -> bool:
    """Whether OCR text is (part of) a signage word, allowing small misreads."""
    return (word in target or target in word
            or difflib.SequenceMatcher(None, word, target).ratio() >= SIGNAGE_MATCH)


class SignageReader:
    """Finds racetrack signage words (e.g. LANGLEY SPEEDWAY) with EasyOCR."""

    def __init__(self, gpu: bool, words: tuple[str, ...]) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.words = words

    def find(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Corner points (4 x 2, image pixels) of each signage word found."""
        boxes = []
        for box, text, confidence in self.reader.readtext(rgb):
            word = "".join(ch for ch in str(text).upper() if ch.isalpha())
            if (confidence >= SIGNAGE_MIN_CONFIDENCE and len(word) >= 4
                    and any(is_signage_word(word, target) for target in self.words)):
                boxes.append(np.asarray(box, np.float32))
        return boxes


def crop_features(image: np.ndarray, plan: CropPlan, signs: list[np.ndarray],
                  car_lit: float) -> tuple[float, float]:
    """(night, signage) scores of one crop plan, each 0 to 1."""
    crop = render_crop(image, plan)
    luminance = crop.mean(axis=2) / 255.0
    dark = float((luminance < NIGHT_DARK_LEVEL).mean())
    lit = min(1.0, car_lit / CAR_LIT_TARGET) if math.isfinite(car_lit) else 0.0
    night = min(1.0, dark / NIGHT_DARK_TARGET) * lit

    height, width = image.shape[:2]
    matrix, _, _ = rotation_matrix(width, height, plan.angle)
    covered = 0.0
    for box in signs:
        points = np.array([transform_point(matrix, tuple(point)) for point in box])
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        area = max(1e-6, (x1 - x0) * (y1 - y0))
        inside_w = max(0.0, min(x1, plan.x + plan.width) - max(x0, plan.x))
        inside_h = max(0.0, min(y1, plan.y + plan.height) - max(y0, plan.y))
        covered += inside_w * inside_h / area
    return night, min(1.0, covered)


def choose_zoom(composition: Composition, image: np.ndarray, clip_model: ClipModel | None,
                signs: list[np.ndarray] | None = None,
                car_lit: float = float("nan")) -> Composition:
    """Keep the tight crop unless a looser one is clearly better.

    ``image`` is the upright analysis-size copy, which the crop plans were
    made on. Each option is rated as its aesthetic score plus NIGHT_BONUS
    times its night score plus SIGN_BONUS times its signage score; a looser
    fill level wins only when it rates at least ZOOM_OUT_MARGIN higher.
    """
    if composition.plan is None:
        return composition
    signs = signs or []

    def rate(option: Composition) -> float:
        option.night, option.signage = crop_features(image, option.plan, signs, car_lit)
        option.crop_aesthetic = (clip_model.aesthetic(render_crop(image, option.plan))
                                 if clip_model is not None else float("nan"))
        aesthetic = option.crop_aesthetic if math.isfinite(option.crop_aesthetic) else 0.0
        return aesthetic + NIGHT_BONUS * option.night + SIGN_BONUS * option.signage

    if composition.rule != "thirds":
        rate(composition)
        return composition
    tight_rating = rate(composition)
    chosen, gain = composition, 0.0
    for alternative in composition.alternatives:
        improvement = rate(alternative) - tight_rating
        if improvement >= ZOOM_OUT_MARGIN and improvement > gain:
            chosen, gain = alternative, improvement
    if chosen is composition:
        chosen.rule = f"thirds, tight ({chosen.fill:.0%} fill)"
    else:
        why = [name for name, delta in (("signage", chosen.signage - composition.signage),
                                        ("night", chosen.night - composition.night))
               if delta > 0.05]
        chosen.rule = (f"thirds, zoomed out to {chosen.fill:.0%} fill (+{gain:.2f} over "
                       f"{composition.fill:.0%}{': ' + ', '.join(why) if why else ''})")
        chosen.candidates = composition.candidates
    return chosen


def compose_candidates(frames: list[Frame], args: argparse.Namespace,
                       clip_model: ClipModel | None,
                       sign_reader: SignageReader | None) -> None:
    """Choose each candidate's tilt and crop, and score the crop's coolness.

    Done before ranking so the ranking and the duplicate check judge the
    photo as it will be saved.
    """
    candidates = [f for f in frames if f.status == "candidate"]
    for index, frame in enumerate(candidates, 1):
        started = time.perf_counter()
        rgb, _, _ = load_upright(frame.path)
        small, _ = downscale(rgb, ANALYSIS_LONG_EDGE)
        width, height = frame.analysis_size
        mask = polygon_mask(frame.subject_polygon, width, height)
        car = small.mean(axis=2)[mask > 0] / 255.0
        frame.car_lit = float(np.percentile(car, 90)) if car.size else float("nan")
        frame.signs = sign_reader.find(small) if sign_reader is not None else []
        composition = compose(mask, frame.driver_point, frame.facing, args.tilt,
                              frame.horizon, frame.width / frame.height, args.max_fill)
        composition = choose_zoom(composition, small, clip_model, frame.signs, frame.car_lit)
        frame.composition = composition
        frame.crop_aesthetic = composition.crop_aesthetic
        frame.night, frame.signage = composition.night, composition.signage
        if clip_model is not None and composition.plan is not None:
            frame.crop_embedding = clip_model.embed(render_crop(small, composition.plan))
        print(f"[{index}/{len(candidates)}] {frame.path.name}  {composition.rule}  "
              f"night {frame.night:.2f}  signage {frame.signage:.2f}  "
              f"({time.perf_counter() - started:.1f}s)")


def load_sign_reader(args: argparse.Namespace) -> SignageReader | None:
    """EasyOCR signage finder, or None when disabled or unavailable."""
    words = tuple(w.strip().upper() for w in args.signage_words.split(",") if w.strip())
    if not words:
        return None
    try:
        gpu = torch_device(args.device).startswith("cuda")
        print(f"Loading signage reader ({', '.join(words)}) ...")
        return SignageReader(gpu, words)
    except Exception as exc:  # torch or easyocr missing, or model download failed
        print(f"Signage reader unavailable ({exc}); signage not scored.")
        return None


def check_nose_up(frame: Frame, composition: Composition) -> None:
    """Assert the rotation raised the nose; a failure means the sign is wrong."""
    if composition.plan is None or composition.rotation == 0:
        return
    assert facing_sign(frame.facing) * composition.rotation > 0, (
        f"{frame.path.name}: rotation {composition.rotation:+.1f} disagrees with facing {frame.facing}")
    if composition.nose_up_reachable:
        assert composition.nose[1] < composition.tail[1], (
            f"{frame.path.name}: nose below tail after rotation; rotation sign is wrong")
    else:
        print(f"  {frame.path.name}: nose is more than {MAX_TILT:.0f} degrees below the tail; "
              "nose-up not reachable")


def output_name(frame: Frame, hero: bool) -> str:
    prefix = frame.driver.lstrip("#") if frame.driver else f"P{frame.group:03d}"
    return f"{prefix}_{frame.path.stem}{'_hero' if hero else ''}.jpg"


def process_keepers(frames: list[Frame], output_dir: Path, args: argparse.Namespace) -> None:
    """Tilt, crop, and save one image for every chosen photo."""
    keepers = [f for f in frames if f.status in ("hero", "keeper")]
    log_path = output_dir / "composition_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for index, frame in enumerate(keepers, 1):
            hero = frame.status == "hero"
            path = output_dir / output_name(frame, hero)
            stale = output_dir / output_name(frame, not hero)
            if stale.exists():
                stale.unlink()  # hero choice changed since an earlier run
            frame.outputs = [path.name]

            composition = frame.composition
            assert composition is not None, f"{frame.path.name}: not composed"
            rgb, exif, icc_profile = load_upright(frame.path)
            check_nose_up(frame, composition)
            frame.compose_rule = composition.rule
            frame.rotation = composition.rotation
            frame.driver_frac = composition.driver_frac
            frame.compose_score = composition.score
            frame.compose_candidates = len(composition.candidates)

            driver_x, driver_y = composition.driver_frac
            summary = (f"{'[best available]  ' if frame.quality == 'best available' else ''}"
                       f"facing {frame.facing} ({frame.facing_source})  {composition.rule}  "
                       f"rotation {composition.rotation:+.1f}°  driver at "
                       f"({driver_x:.2f}, {driver_y:.2f}) [{frame.driver_source}]")
            if math.isfinite(composition.score):
                summary += (f"  winner {composition.score:.2f} of "
                            f"{len(composition.candidates)} candidates")
            log.write(json.dumps({
                "file": frame.path.name, "output": path.name, "facing": frame.facing,
                "facing_source": frame.facing_source, "rule": composition.rule,
                "rotation": round(composition.rotation, 1),
                "driver": [round(driver_x, 3), round(driver_y, 3)],
                "driver_source": frame.driver_source,
                "score": None if not math.isfinite(composition.score) else round(composition.score, 3),
                "fill": None if not math.isfinite(composition.fill) else composition.fill,
                "candidates": composition.candidates,
            }) + "\n")

            if path.exists() and not args.force:
                print(f"[{index}/{len(keepers)}] {path.name}  already processed; {summary}")
                continue
            output = render_crop(rgb, composition.plan) if composition.plan else rgb
            save_jpeg(output, path, exif, icc_profile, args.quality)
            print(f"[{index}/{len(keepers)}] {path.name}  {summary}")

    names = {name for frame in keepers for name in frame.outputs}
    assert len(names) == len(keepers), f"{len(names)} outputs for {len(keepers)} photos"


# ---------------------------------------------------------------------------
# Catalog and reporting
# ---------------------------------------------------------------------------

CATALOG_FIELDS = [
    "file", "capture_time", "time_source", "pass", "driver", "pass_number",
    "match_similarity", "facing", "facing_source", "roof_offset", "end_offset",
    "status", "reason", "rank",
    "score", "picked", "keep", "vehicle", "vehicle_confidence", "car_sharpness",
    "bg_sharpness",
    "blur", "display_blur", "car_blur_small", "bg_blur_small", "quality",
    "crop_aesthetic", "night", "signage",
    "blur_ratio", "pan_coherence", "subject_area", "wheels", "wheel_blur", "attitude",
    "aesthetic", "car_number", "number_score", "clip_fraction", "people", "drift_x",
    "drift_response", "horizon", "driver_source", "compose_rule", "rotation",
    "driver_x_frac", "driver_y_frac", "compose_score", "compose_candidates", "outputs",
]


def write_catalog(frames: list[Frame], numbers: dict[int, str],
                  similarities: dict[int, float], path: Path) -> None:
    """Write one CSV row per frame with every measurement and decision."""
    def number(value: float, digits: int = 4) -> str:
        return "" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CATALOG_FIELDS)
        for f in frames:
            writer.writerow([
                f.path.name,
                datetime.fromtimestamp(f.capture_time).isoformat(timespec="milliseconds"),
                f.time_source, f.group, f.driver, numbers.get(f.group, ""),
                number(similarities.get(f.group, float("nan")), 3),
                f.facing, f.facing_source, number(f.roof_offset, 3), number(f.end_offset, 3),
                f.status, f.reason, f.rank or "", number(f.score), "yes" if f.picked else "",
                "yes" if f.keep else "",
                f.subject_class, number(f.subject_confidence, 3),
                number(f.car_sharpness, 2), number(f.bg_sharpness, 2), number(f.blur, 3),
                number(f.display_blur, 3), number(f.car_blur_small, 3),
                number(f.bg_blur_small, 3), f.quality,
                number(f.crop_aesthetic, 3), number(f.night, 3), number(f.signage, 3),
                number(f.blur_ratio, 3), number(f.pan_coherence, 3),
                number(f.subject_area), f.wheel_count, number(f.wheel_blur, 3),
                number(f.attitude, 2), number(f.aesthetic, 3),
                f.car_number, number(f.number_score, 3),
                number(f.clip_fraction), f.people,
                number(f.drift_x, 1), number(f.drift_response, 3),
                number(f.horizon, 1), f.driver_source, f.compose_rule, number(f.rotation, 1),
                number(f.driver_frac[0], 3), number(f.driver_frac[1], 3),
                number(f.compose_score, 3), f.compose_candidates or "",
                ";".join(f.outputs),
            ])


def print_summary(frames: list[Frame], output_dir: Path) -> None:
    status_counts = Counter(f.status for f in frames)
    reasons = Counter(f.reason.split(" (")[0] for f in frames if f.status == "rejected")
    passes = len({f.group for f in frames})
    drivers = len({f.driver for f in frames if f.driver})
    print(f"\n{len(frames)} frames, {passes} passes, {drivers} drivers: "
          f"{status_counts['hero']} heroes, {status_counts['keeper']} other keepers, "
          f"{status_counts['passed']} passed over, {status_counts['rejected']} rejected")
    for reason, count in reasons.most_common():
        print(f"  rejected, {reason}: {count}")
    print(f"Output: {output_dir}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def unit_interval(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cull motorsport photos, pick each car's hero frame, "
                    "and export a tilted rule-of-thirds crop of every keeper.")
    parser.add_argument("folder", type=Path, help="folder of JPEGs from one shoot")
    parser.add_argument("--output", type=Path,
                        help="output folder (default: <folder>/processed)")

    grouping = parser.add_argument_group("grouping and selection")
    grouping.add_argument("--gap", type=float, default=12.0,
                          help="seconds between frames that always start a new pass; "
                               "shorter gaps also split when the car changes (default 12)")
    grouping.add_argument("--select-by", choices=["driver", "pass"], default="driver",
                          help="pick one hero per driver or per pass (default driver)")
    grouping.add_argument("--per-driver", type=int, default=PHOTOS_PER_DRIVER,
                          help="most photos exported per driver, best and most varied "
                               f"first (default {PHOTOS_PER_DRIVER})")
    grouping.add_argument("--duplicate-similarity", type=unit_interval,
                          default=DUPLICATE_SIMILARITY,
                          help="crops at least this alike (0 to 1) count as the same shot; "
                               f"only the better is exported (default {DUPLICATE_SIMILARITY})")
    grouping.add_argument("--max-blur", type=unit_interval, default=0.5,
                          help="frames whose vehicle blur score is above this are kept "
                               "only as a driver's best available photo; about 0.3 is "
                               "crisp, see the catalog's blur column (default 0.5)")
    grouping.add_argument("--max-display-blur", type=unit_interval, default=0.70,
                          help="the same for blur as a viewer sees it: the car enlarged "
                               "as the crop does, noise removed, strongest edges judged; "
                               "about 0.5 to 0.6 is crisp (default 0.70)")
    grouping.add_argument("--relative-sharpness", type=unit_interval, default=0.6,
                          help="reject frames whose vehicle sharpness is below this "
                               "fraction of the pass's sharpest frame (default 0.6)")
    grouping.add_argument("--sharpness-floor", type=float, default=0.0,
                          help="absolute minimum vehicle sharpness; calibrate from "
                               "the catalog's car_sharpness column (default 0, off)")

    styling = parser.add_argument_group("tilt and crop")
    styling.add_argument("--tilt", type=float, default=10.0,
                         help="typical nose-up tilt in degrees; each shot tries 0.6x, 1x, "
                              "and 1.4x, more if needed to raise the nose; 0 disables "
                              "(default 10)")
    styling.add_argument("--direction", choices=["auto", "left", "right"], default="auto",
                         help="which way cars face: auto-detect per photo, or force for all")
    styling.add_argument("--default-direction", choices=["right", "left"], default="right",
                         help="direction used when auto-detection is inconclusive")
    styling.add_argument("--max-fill", type=unit_interval, default=DEFAULT_MAX_FILL,
                         help="tightest crop: largest share of the crop width the car "
                              f"may fill, up to {FILL_LEVELS[0]} (default {DEFAULT_MAX_FILL})")
    styling.add_argument("--signage-words", default=",".join(SIGNAGE_WORDS),
                         help="comma-separated track signage words that make a crop "
                              "cooler; empty disables (default "
                              f"{','.join(SIGNAGE_WORDS)})")
    styling.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")

    model = parser.add_argument_group("detection model")
    model.add_argument("--model", default="yolo11s-seg.pt",
                       help="Ultralytics segmentation weights (default yolo11s-seg.pt)")
    model.add_argument("--conf", type=unit_interval, default=0.35,
                       help="detection confidence threshold (default 0.35)")
    model.add_argument("--imgsz", type=int, default=960,
                       help="model input size in pixels (default 960)")
    model.add_argument("--device", default=None,
                       help="inference device, e.g. cpu, 0, or mps (default: automatic)")

    identity = parser.add_argument_group("aesthetics and driver identification")
    identity.add_argument("--no-clip", action="store_true",
                          help="skip aesthetic scoring and appearance matching")
    identity.add_argument("--number-reader", choices=["vlm", "easyocr"], default="vlm",
                          help="read car numbers with a vision-language model via Ollama "
                               "(falls back to EasyOCR if unreachable) or with EasyOCR "
                               "(default vlm)")
    identity.add_argument("--ollama-url", default=OLLAMA_URL,
                          help=f"Ollama server for number reading (default {OLLAMA_URL})")
    identity.add_argument("--ollama-model", default=OLLAMA_MODEL,
                          help=f"vision model for number reading (default {OLLAMA_MODEL})")
    identity.add_argument("--no-ocr", action="store_true",
                          help="skip car-number reading")
    identity.add_argument("--match-threshold", type=unit_interval, default=0.92,
                          help="appearance similarity needed to join an unnumbered "
                               "pass to another pass (default 0.92)")
    identity.add_argument("--embed-top", type=int, default=3,
                          help="sharpest frames per pass used for appearance (default 3)")

    learning = parser.add_argument_group("ranking weights")
    learning.add_argument("--weights", type=Path,
                          help="JSON file of ranking weights, e.g. learned_weights.json")
    learning.add_argument("--learn-from", type=Path,
                          help="folder of your hand-picked photos from this shoot; fits "
                               "and saves ranking weights instead of exporting images")

    parser.add_argument("--force", action="store_true",
                        help="reprocess frames whose output already exists")
    parser.add_argument("--catalog-only", action="store_true",
                        help="analyze and write the catalog without exporting images")

    args = parser.parse_args(argv)
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if args.per_driver < 1:
        parser.error("--per-driver must be at least 1")
    if args.embed_top < 1:
        parser.error("--embed-top must be at least 1")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"--weights file not found: {args.weights}")
    if args.learn_from is not None and not args.learn_from.is_dir():
        parser.error(f"--learn-from folder not found: {args.learn_from}")
    return args


def stay_offline_when_cached(args: argparse.Namespace) -> None:
    """Skip network calls for models that are already on disk.

    Hugging Face re-checks cached files every time CLIP loads, and Ultralytics
    sends usage analytics while it considers itself online. Neither is needed
    once the weights are downloaded, so both are switched off for this run.
    A model that is not downloaded yet still downloads on first use. Photos
    are never sent anywhere; number reading uses the local Ollama server.
    """
    hub_cache = Path(os.environ.get("HF_HUB_CACHE")
                     or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
                     / "hub")
    if (hub_cache / CLIP_HF_CACHE_NAME).is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if Path(args.model).is_file():
        os.environ.setdefault("YOLO_OFFLINE", "1")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stay_offline_when_cached(args)
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2
    output_dir = (args.output or folder / "processed").expanduser().resolve()

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in JPEG_EXTENSIONS)
    if not files:
        print(f"No JPEG files found in {folder}", file=sys.stderr)
        return 1

    frames = [Frame(path, *read_capture_time(path)) for path in files]
    frames.sort(key=lambda f: (f.capture_time, f.path.name))
    assign_groups(frames, args.gap)
    keep = read_keep_list(folder)
    for frame in frames:
        frame.keep = frame.path.stem.lower() in keep
    if keep:
        found = sum(f.keep for f in frames)
        print(f"Keep list: {found} of {len(keep)} listed photos found.")
    if any(f.time_source == "file" for f in frames):
        print("Note: some files lack EXIF capture time; file times were used instead.")

    clip_model = load_clip(args)
    print(f"Loading model {args.model} ...")
    detector = VehicleDetector(args.model, args.conf, args.imgsz, args.device)
    analyze_all(frames, detector, args, clip_model)
    if clip_model is not None:
        print(f"Passes split where the car changes: {split_passes(frames, args.gap)}")

    passes: dict[int, list[Frame]] = defaultdict(list)
    for frame in frames:
        passes[frame.group].append(frame)
    for members in passes.values():
        reject_in_pass(members, args)

    print("\nScoring candidates ...")
    number_reader = load_reader(args)
    enrich_candidates(frames, args, clip_model, number_reader)

    labels, numbers, similarities = identify_drivers(passes, args.match_threshold)
    for frame in frames:
        frame.driver = labels[frame.group]

    for label, members in ranking_groups(frames, args.select_by).items():
        drop_weak_frames(members, args, label)
        drop_soft_duplicates(members)

    print("\nComposing candidates ...")
    compose_candidates(frames, args, clip_model, load_sign_reader(args))

    weights = load_weights(args.weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.learn_from is not None:
        learn_weights(frames, weights, args, output_dir)

    for label, members in ranking_groups(frames, args.select_by).items():
        rank_group(members, weights, label, args.per_driver, args.duplicate_similarity)

    if not args.catalog_only and args.learn_from is None:
        print("\nExporting ...")
        process_keepers(frames, output_dir, args)

    write_catalog(frames, numbers, similarities, output_dir / "catalog.csv")
    print_summary(frames, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
