"""Synthetic checks of facing detection, thirds composition, and nose-up tilt."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import motorsport_cull as mc  # noqa: E402

WIDTH, HEIGHT = 1600, 1067
ASPECT = WIDTH / HEIGHT


def synthetic_car(facing: str, center=(800, 560), length=560, nose_drop=0.0):
    """Side-view car silhouette with the cabin toward the tail.

    Returns (mask, driver point, nose x). ``nose_drop`` tilts the car
    nose-down by that many degrees, as perspective does in some shots.
    """
    cx, cy = center
    half = length / 2
    body_h, cabin_h = 0.2 * length, 0.16 * length
    # Built facing right: tail at the left, nose at the right.
    body = [(-half, 0), (half, 0), (half, -body_h), (-half, -body_h)]
    cabin = [(-0.38 * half, -body_h), (0.30 * half, -body_h),
             (0.05 * half, -body_h - cabin_h), (-0.30 * half, -body_h - cabin_h)]
    driver = (0.0, -body_h - 0.45 * cabin_h)
    sign = 1 if facing == "right" else -1
    drop = np.tan(np.radians(nose_drop))

    def place(x, y):
        # Shear the nose down, mirror for left-facing, move into the frame.
        return cx + sign * x, cy + y + x * drop

    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for shape in (body, cabin):
        points = np.array([place(x, y) for x, y in shape], np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask, place(*driver), place(half, -body_h / 2)[0]


@pytest.mark.parametrize("facing", ["left", "right"])
def test_roofline_detects_facing(facing):
    mask, _, _ = synthetic_car(facing)
    offset = mc.roof_offset(mask)
    assert abs(offset) >= mc.FACING_MIN_OFFSET
    assert ("left" if offset > 0 else "right") == facing


@pytest.mark.parametrize("facing", ["left", "right"])
def test_taller_end_is_tail_in_head_on_view(facing):
    # Near head-on: symmetric roof, flank on the tail side standing taller.
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    nose_h, tail_h = 200, 260
    if facing == "left":
        outline = [(600, 700), (1000, 700), (1000, 700 - tail_h), (600, 700 - nose_h)]
    else:
        outline = [(600, 700), (1000, 700), (1000, 700 - nose_h), (600, 700 - tail_h)]
    cv2.fillPoly(mask, [np.array(outline, np.int32)], 255)
    offset = mc.end_offset(mask)
    assert abs(offset) >= mc.END_MIN_OFFSET
    assert ("left" if offset > 0 else "right") == facing


@pytest.mark.parametrize("nose_drop", [0.0, 12.0])
@pytest.mark.parametrize("facing", ["left", "right"])
def test_tight_crop_lead_space_and_nose_up(facing, nose_drop):
    mask, driver, _ = synthetic_car(facing, nose_drop=nose_drop)
    result = mc.compose(mask, driver, facing, tilt=10.0, horizon=float("nan"), aspect=ASPECT)

    assert result.rule == "thirds"
    plan = result.plan
    driver_x, driver_y = result.driver_frac

    # Driver pulled toward the thirds line on the tail side of the frame.
    assert driver_x < 0.5 if facing == "right" else driver_x > 0.5

    # Lead space: the strip ahead of the nose holds no car pixels.
    matrix, cw, ch = mc.rotation_matrix(WIDTH, HEIGHT, plan.angle)
    rotated = cv2.warpAffine(mask, matrix, (cw, ch), flags=cv2.INTER_NEAREST)
    crop = rotated[plan.y:plan.y + plan.height, plan.x:plan.x + plan.width]
    lead_width = int(mc.LEAD_MIN * plan.width) - 2
    lead = crop[:, -lead_width:] if facing == "right" else crop[:, :lead_width]
    assert not lead.any(), "car intrudes into the lead space"
    # The whole car is inside the crop.
    assert crop.sum() == rotated.sum()

    # Tight: the car fills most of the width, tail close to the edge.
    columns = np.flatnonzero(crop.any(axis=0))
    car_fill = (columns[-1] - columns[0] + 1) / plan.width
    tail_gap = columns[0] if facing == "right" else plan.width - 1 - columns[-1]
    assert car_fill >= mc.DEFAULT_MAX_FILL - 0.06
    assert tail_gap <= 0.05 * plan.width

    # Nose-up: rotation sign follows facing, and the nose ends above the tail.
    assert mc.facing_sign(facing) * result.rotation > 0
    assert result.nose[1] < result.tail[1]
    # Right-facing rotates counterclockwise (positive), left-facing clockwise.
    assert (result.rotation > 0) == (facing == "right")


@pytest.mark.parametrize("facing", ["left", "right"])
def test_candidates_vary_fill_and_tilt(facing):
    mask, driver, _ = synthetic_car(facing, length=420)
    result = mc.compose(mask, driver, facing, tilt=10.0, horizon=0.0, aspect=ASPECT)
    fills = {c["fill"] for c in result.candidates}
    tilts = {abs(c["rotation"]) for c in result.candidates}
    assert len(fills) >= 2
    assert len(tilts) >= 2
    # The tightest fill that fits wins; looser ones remain as alternatives.
    assert result.fill == max(fills)
    assert all(alt.fill < result.fill for alt in result.alternatives)
    assert round(result.score, 3) == max(
        c["score"] for c in result.candidates if c["fill"] == result.fill)


def test_max_fill_caps_tightness():
    mask, driver, _ = synthetic_car("right", length=420)
    result = mc.compose(mask, driver, "right", tilt=10.0, horizon=0.0, aspect=ASPECT,
                        max_fill=0.75)
    assert result.fill <= 0.75


def test_centered_fallback_when_thirds_cannot_fit():
    # A car filling most of the frame leaves no room for lead space.
    mask, driver, _ = synthetic_car("right", length=1500)
    result = mc.compose(mask, driver, "right", tilt=10.0, horizon=float("nan"), aspect=ASPECT)
    assert result.rule != "thirds"


def test_reblur_separates_sharp_from_blurred_regardless_of_texture():
    rng = np.random.default_rng(0)
    region = np.ones((300, 600), bool)
    for density in (0.02, 0.3):   # plain panels and a busy livery
        texture = (rng.random((300, 600)) < density).astype(np.float32) * 255
        sharp = cv2.GaussianBlur(texture, (0, 0), 0.7)
        focus_miss = cv2.GaussianBlur(texture, (0, 0), 4)
        # Motion blur along one axis only; camera motion blur has soft ends,
        # unlike a hard box filter.
        pan_miss = cv2.GaussianBlur(texture, (0, 0), sigmaX=6, sigmaY=0.7)
        assert mc.reblur_score(sharp, region) < 0.4
        assert mc.reblur_score(focus_miss, region) > 0.5
        assert mc.reblur_score(pan_miss, region) > 0.5


def make_frame(name, blur, facing="right", aspect=3.0, keep=False):
    frame = mc.Frame(Path(f"{name}.JPG"), 0.0, "exif")
    frame.status, frame.blur, frame.facing, frame.keep = "candidate", blur, facing, keep
    frame.subject_polygon = np.array([[0, 0], [100 * aspect, 0], [100 * aspect, 100], [0, 100]],
                                     np.float32)
    return frame


def test_soft_frame_dropped_only_when_a_crisp_frame_covers_its_view():
    crisp_side = make_frame("crisp_side", 0.31)
    soft_same_view = make_frame("soft_same_view", 0.40)
    soft_other_facing = make_frame("soft_other_facing", 0.40, facing="left")
    soft_angled = make_frame("soft_angled", 0.40, aspect=1.6)
    soft_kept = make_frame("soft_kept", 0.40, keep=True)
    frames = [crisp_side, soft_same_view, soft_other_facing, soft_angled, soft_kept]
    mc.drop_soft_duplicates(frames)
    assert [f.status for f in frames] == [
        "candidate", "rejected", "candidate", "candidate", "candidate"]
    assert "crisp_side.JPG" in soft_same_view.reason


def test_soft_frames_kept_when_driver_has_no_crisp_frame():
    frames = [make_frame("a", 0.42), make_frame("b", 0.45)]
    mc.drop_soft_duplicates(frames)
    assert all(f.status == "candidate" for f in frames)


def test_keep_list_reads_names_stems_and_comments(tmp_path):
    (tmp_path / mc.KEEP_LIST_NAME).write_text(
        "IMG_6517.JPG\n  img_6600  # great pan\n# a comment line\n\n", encoding="utf-8")
    assert mc.read_keep_list(tmp_path) == {"img_6517", "img_6600"}
    assert mc.read_keep_list(tmp_path / "missing") == set()


class FakeReader:
    def read(self, rgb):
        return "42", 0.2


def test_vision_reader_switches_to_fallback_when_server_stops_answering(monkeypatch):
    reader = object.__new__(mc.VisionNumberReader)
    reader.url, reader.model, reader.gpu = "http://127.0.0.1:9", "test-model", False  # closed port
    reader.fallback, reader.failed = None, False
    monkeypatch.setattr(mc, "OLLAMA_RETRY_WAITS", (0, 0))   # no real waiting in the test
    image = np.zeros((64, 64, 3), np.uint8)
    # The first read fails over the network; it and every later read use EasyOCR.
    original = mc.NumberReader
    mc.NumberReader = lambda gpu: FakeReader()
    try:
        assert reader.read(image) == ("42", 0.2)
        assert reader.failed
        assert reader.read(image) == ("42", 0.2)
    finally:
        mc.NumberReader = original


def detection(class_id, width, height):
    box = np.array([0, 0, width, height], np.float32)
    outline = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    return mc.Detection(class_id, 0.9, box, outline)


def test_car_on_course_preferred_over_larger_parked_truck():
    truck, car = detection(7, 200, 100), detection(2, 120, 60)   # car is 36% of the truck
    assert mc.choose_subject([truck, car]) is car


def test_truck_kept_when_no_comparable_car():
    truck, speck = detection(7, 200, 100), detection(2, 40, 20)  # car is 4% of the truck
    assert mc.choose_subject([truck, speck]) is truck
    assert mc.choose_subject([truck]) is truck
    assert mc.choose_subject([]) is None


BLUR_ARGS = __import__("argparse").Namespace(max_blur=0.5, max_display_blur=0.70)


def blur_frame(name, blur=0.33, display=0.60, car=0.32, bg=0.30, keep=False):
    frame = make_frame(name, blur, keep=keep)
    frame.display_blur, frame.car_blur_small, frame.bg_blur_small = display, car, bg
    return frame


def test_missed_pan_and_unusable_blur_are_always_rejected():
    assert "sharp background" in mc.hard_blur_reason(blur_frame("pan", car=0.55, bg=0.26))
    assert "unusably" in mc.hard_blur_reason(blur_frame("smear", blur=0.64))
    assert mc.hard_blur_reason(blur_frame("fine")) == ""
    # A soft car on an equally soft background is not a missed pan.
    assert mc.hard_blur_reason(blur_frame("shake", car=0.50, bg=0.45)) == ""


def test_blurry_frames_dropped_when_driver_has_a_sharp_one():
    sharp = blur_frame("sharp")
    enlarged = blur_frame("enlarged", display=0.75)      # soft only at display size
    smeared = blur_frame("smeared", car=0.45, bg=0.31)    # moderate pan gap
    kept = blur_frame("kept", display=0.80, keep=True)
    frames = [sharp, enlarged, smeared, kept]
    mc.drop_weak_frames(frames, BLUR_ARGS, "#1")
    assert [f.status for f in frames] == ["candidate", "rejected", "rejected", "candidate"]
    assert "display size" in enlarged.reason and "better photos of #1" in enlarged.reason
    assert sharp.quality == "good"


def test_driver_without_a_sharp_frame_keeps_its_best():
    worse, better = blur_frame("worse", display=0.80), blur_frame("better", display=0.74)
    mc.drop_weak_frames([worse, better], BLUR_ARGS, "#2")
    assert better.status == "candidate" and better.quality == "best available"
    assert worse.status == "rejected"


def test_signage_words_match_despite_misreads_and_fragments():
    assert mc.is_signage_word("LANGLEY", "LANGLEY")
    assert mc.is_signage_word("SPEEDWA", "SPEEDWAY")    # cut off at the frame edge
    assert mc.is_signage_word("LANGIEY", "LANGLEY")     # misread letter
    assert not mc.is_signage_word("BOJANGLES", "SPEEDWAY")
    assert not mc.is_signage_word("GRANITE", "LANGLEY")


def test_crop_features_score_night_and_signage():
    image = np.full((400, 600, 3), 200, np.uint8)
    image[:150] = 5                                          # dark night sky
    sign = np.array([[100, 40], [300, 40], [300, 80], [100, 80]], np.float32)
    with_sign = mc.CropPlan(0.0, 50, 0, 400, 267, 600, 400)
    without_sign = mc.CropPlan(0.0, 50, 133, 400, 267, 600, 400)
    night, signage = mc.crop_features(image, with_sign, [sign], car_lit=0.8)
    assert night == 1.0 and signage == 1.0
    night, signage = mc.crop_features(image, without_sign, [sign], car_lit=0.8)
    assert night < 0.3 and signage == 0.0      # only a sliver of sky left
    night, _ = mc.crop_features(image, with_sign, [sign], car_lit=0.1)   # car in the dark
    assert night < 0.3


def selection_frame(name, score, embedding, keep=False):
    frame = make_frame(name, 0.3, keep=keep)
    frame.car_sharpness = score
    frame.crop_embedding = np.asarray(embedding, np.float32) / np.linalg.norm(embedding)
    return frame


def test_selection_caps_per_driver_and_skips_near_identical_shots():
    weights = {"car_sharpness": 1.0}
    frames = [selection_frame("best", 5, [1, 0, 0]),
              selection_frame("same_shot", 4, [1, 0.01, 0]),   # near-identical to best
              selection_frame("other_angle", 3, [0, 1, 0]),
              selection_frame("third_angle", 2, [0, 0, 1]),
              selection_frame("kept_anyway", 1, [1, 0.02, 0], keep=True)]
    mc.rank_group(frames, weights, "#7", per_driver=2, duplicate_similarity=0.94)
    status = {f.path.stem: f.status for f in frames}
    assert status == {"best": "hero", "same_shot": "passed", "other_angle": "keeper",
                      "third_angle": "passed", "kept_anyway": "keeper"}
    assert "near-identical to best.JPG" in frames[1].reason
    assert "already has 2 photos" in frames[3].reason


def test_display_blur_sees_through_night_noise():
    # Lettering-like detail; at night it is buried in sensor noise.
    rng = np.random.default_rng(1)
    car = np.full((300, 900), 60, np.float32)
    for x in range(60, 840, 70):
        cv2.rectangle(car, (x, 90), (x + 30, 210), 220, -1)
    smeared = cv2.blur(car, (9, 1))                          # pan did not quite track the car
    noisy = lambda img: np.clip(img + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)
    outline = np.array([[0, 0], [900, 0], [900, 300], [0, 300]], np.float32)
    as_rgb = lambda img: np.dstack([img] * 3)
    sharp_score = mc.display_blur_score(as_rgb(noisy(car)), outline)
    smeared_score = mc.display_blur_score(as_rgb(noisy(smeared)), outline)
    assert smeared_score - sharp_score > 0.1
    # Without denoising and strong-edge selection, noise makes both look equally sharp.
    region = np.ones(car.shape, bool)
    plain = [mc.reblur_score(noisy(img).astype(np.float32), region) for img in (car, smeared)]
    assert abs(plain[1] - plain[0]) < 0.05


def timed_frame(seconds, embedding):
    frame = mc.Frame(Path(f"t{seconds}.JPG"), float(seconds), "exif")
    if embedding is not None:
        vector = np.asarray(embedding, np.float32)
        frame.car_embedding = vector / np.linalg.norm(vector)
    return frame


def test_passes_split_where_the_car_changes_not_just_on_long_gaps():
    red, blue = [1, 0.1, 0], [0.1, 1, 0]
    frames = [timed_frame(0.0, red), timed_frame(0.4, red), timed_frame(0.8, red),
              timed_frame(4.0, blue),           # short gap, different car: new pass
              timed_frame(4.4, blue),
              timed_frame(4.7, None),           # no car detected: stays with its pass
              timed_frame(5.0, blue),
              timed_frame(5.2, red),            # different car but no real gap (burst)
              timed_frame(30.0, red)]           # long gap: always a new pass
    count = mc.split_passes(frames, gap_seconds=12)
    assert [f.group for f in frames] == [0, 0, 0, 1, 1, 1, 1, 1, 2]
    assert count == 3
