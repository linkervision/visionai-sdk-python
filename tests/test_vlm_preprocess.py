"""Tests for VLM image-resize planning helpers."""

import pytest

from visionai_sdk_python.vlm import RESIZE_OPTIONS, compute_resize

# Qwen2-VL reference values, kept as local literals (not public API).
QWEN_FACTOR = 28
QWEN_MAX_PIXELS = 14 * 14 * 4 * 1280
QWEN_DEFAULT_MIN_PIXELS = 4 * 28 * 28  # == 56 * 56


@pytest.mark.parametrize(
    "height,width",
    [
        (1080, 1920),  # downscale (over max_pixels)
        (28, 28),  # tiny, upscale to min_pixels
        (500, 375),  # within bounds, snap to factor
        (3000, 4000),  # large downscale
    ],
)
def test_smart_constraints(height, width):
    plan = compute_resize(height, width, factor=QWEN_FACTOR, max_pixels=QWEN_MAX_PIXELS)
    assert plan.width % QWEN_FACTOR == 0
    assert plan.height % QWEN_FACTOR == 0
    assert QWEN_DEFAULT_MIN_PIXELS <= plan.width * plan.height <= QWEN_MAX_PIXELS


def test_smart_keeps_aspect_ratio():
    plan = compute_resize(1080, 1920, factor=QWEN_FACTOR, max_pixels=QWEN_MAX_PIXELS)
    assert plan.width / plan.height == pytest.approx(1920 / 1080, rel=0.1)


def test_smart_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        compute_resize(10, 2010, factor=QWEN_FACTOR, max_pixels=QWEN_MAX_PIXELS)


def test_smart_rejects_unsatisfiable_max_pixels():
    # Elongated strip: aspect ratio passes the <=200 check, but the factor
    # floor cannot fit the pixel budget. Must raise, not silently overshoot.
    with pytest.raises(ValueError, match="cannot satisfy"):
        compute_resize(200, 40000, **RESIZE_OPTIONS["smart_384_p48"])


def test_smart_custom_max_pixels():
    plan = compute_resize(1080, 1920, factor=28, max_pixels=640 * 640)
    assert plan.width * plan.height <= 640 * 640
    assert plan.width % 28 == 0
    assert plan.height % 28 == 0


def test_smart_min_pixels_defaults_to_qwen_convention():
    plan = compute_resize(28, 28, factor=28, max_pixels=QWEN_MAX_PIXELS)
    assert plan.width * plan.height >= QWEN_DEFAULT_MIN_PIXELS
    explicit = compute_resize(
        28,
        28,
        factor=28,
        min_pixels=QWEN_DEFAULT_MIN_PIXELS,
        max_pixels=QWEN_MAX_PIXELS,
    )
    assert plan == explicit


def test_smart_requires_factor_and_max_pixels():
    with pytest.raises(ValueError, match="factor and max_pixels"):
        compute_resize(1080, 1920, factor=28)
    with pytest.raises(ValueError, match="factor and max_pixels"):
        compute_resize(1080, 1920, max_pixels=640 * 640)


def test_smart_rejects_min_pixels_over_max_pixels():
    with pytest.raises(ValueError, match="min_pixels"):
        compute_resize(28, 28, factor=28, min_pixels=100000, max_pixels=1000)
    # Implicit default min_pixels (4 * 48 * 48 = 9216) exceeds max_pixels.
    with pytest.raises(ValueError, match="min_pixels"):
        compute_resize(1080, 1920, factor=48, max_pixels=64 * 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factor": 0, "max_pixels": 1000},
        {"factor": -28, "max_pixels": 1000},
        {"factor": 28, "max_pixels": 0},
        {"factor": 28, "max_pixels": 1000, "min_pixels": 0},
    ],
)
def test_smart_rejects_non_positive_params(kwargs):
    with pytest.raises(ValueError):
        compute_resize(1080, 1920, **kwargs)


@pytest.mark.parametrize("height,width", [(0, 1920), (1920, 0), (0, 0), (-1, 100)])
def test_rejects_non_positive_dimensions(height, width):
    with pytest.raises(ValueError, match="height and width"):
        compute_resize(height, width, longest_edge=512)
    with pytest.raises(ValueError, match="height and width"):
        compute_resize(height, width, factor=28, max_pixels=1000)


def test_fixed_size():
    plan = compute_resize(1080, 1920, size=(448, 224))
    assert (plan.width, plan.height) == (448, 224)


def test_square():
    plan = compute_resize(1080, 1920, square=448, resampling="lanczos")
    assert (plan.width, plan.height) == (448, 448)
    assert plan.resampling == "lanczos"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"square": 0},
        {"square": -448},
        {"size": (0, 224)},
        {"size": (448, 0)},
        {"size": (-5, 3)},
        {"longest_edge": 0},
        {"longest_edge": -10},
    ],
)
def test_rejects_non_positive_targets(kwargs):
    with pytest.raises(ValueError, match="positive"):
        compute_resize(1080, 1920, **kwargs)


@pytest.mark.parametrize(
    "height,width,expected_w,expected_h",
    [
        (1080, 1920, 1024, 576),  # downscale, landscape
        (1920, 1080, 576, 1024),  # downscale, portrait
    ],
)
def test_longest_edge_downscales(height, width, expected_w, expected_h):
    plan = compute_resize(height, width, longest_edge=1024)
    assert (plan.width, plan.height) == (expected_w, expected_h)


def test_longest_edge_never_upscales():
    plan = compute_resize(256, 512, longest_edge=1024)
    assert (plan.width, plan.height) == (512, 256)
    plan = compute_resize(240, 320, **RESIZE_OPTIONS["longest_768"])
    assert (plan.width, plan.height) == (320, 240)


def test_rejects_invalid_resampling():
    with pytest.raises(ValueError, match="resampling"):
        compute_resize(1080, 1920, square=448, resampling="nearest")


def test_rejects_no_mode():
    with pytest.raises(ValueError, match="exactly one mode"):
        compute_resize(1080, 1920)


def test_rejects_multiple_modes():
    with pytest.raises(ValueError, match="exactly one mode"):
        compute_resize(1080, 1920, square=448, longest_edge=1024)
    with pytest.raises(ValueError, match="exactly one mode"):
        compute_resize(1080, 1920, square=448, max_pixels=640 * 640)


@pytest.mark.parametrize("height,width", [(1080, 1920), (480, 640)])
def test_resize_options_all_valid(height, width):
    for key, kwargs in RESIZE_OPTIONS.items():
        plan = compute_resize(height, width, **kwargs)
        assert plan.width > 0 and plan.height > 0, key
        assert plan.resampling == kwargs["resampling"], key
        if "factor" in kwargs:
            assert plan.width % kwargs["factor"] == 0, key
            assert plan.height % kwargs["factor"] == 0, key
            assert plan.width * plan.height <= kwargs["max_pixels"], key


def test_resize_options_immutable():
    with pytest.raises(TypeError):
        RESIZE_OPTIONS["square_384"]["square"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        RESIZE_OPTIONS["square_999"] = {"square": 999}  # type: ignore[index]


def test_smart_rejects_unsatisfiable_min_pixels():
    # Codex P2: tight [min, max] band + extreme shape floors below min_pixels.
    with pytest.raises(ValueError, match="cannot satisfy"):
        compute_resize(10000, 20000, factor=28, max_pixels=3200)
