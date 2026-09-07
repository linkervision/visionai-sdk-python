"""Tests for VLM image-resize planning helpers."""

import pytest

from visionai_sdk_python.vlm import compute_resize


def smart(width, height, **overrides):
    kwargs = {
        "width": width,
        "height": height,
        "name": "smart_resize",
        "factor": 32,
        "pixels": 768,
        "interpolation": "bicubic",
    }
    kwargs.update(overrides)
    return compute_resize(**kwargs)


@pytest.mark.parametrize("factor", [32, 48])
@pytest.mark.parametrize(
    "width,height",
    [
        (1920, 1080),  # downscale (over the pixel budget)
        (32, 32),  # tiny, upscale to min_pixels
        (375, 500),  # within bounds, snap to factor
        (4000, 3000),  # large downscale
    ],
)
def test_smart_constraints(width, height, factor):
    plan = smart(width, height, factor=factor)
    assert plan.width % factor == 0
    assert plan.height % factor == 0
    assert 4 * factor * factor <= plan.width * plan.height <= 768 * 768


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1920, 1080, (1024, 576)),  # README example
        (1280, 8, (1280, 32)),  # thin strip: short side snaps up, long side kept
        (1000, 17, (992, 32)),  # ...and does not jump at the factor/2 boundary
        (1000, 16, (992, 32)),
        (224, 2, (224, 32)),  # already inside the budget — no upscale
    ],
)
def test_smart_golden_dimensions(width, height, expected):
    # Pins parity with the model server's smart_resize; the constraint-only
    # tests above pass for wrong-but-legal plans.
    plan = smart(width, height)
    assert (plan.width, plan.height) == expected


def test_smart_keeps_aspect_ratio():
    plan = smart(1920, 1080)
    assert plan.width / plan.height == pytest.approx(1920 / 1080, rel=0.1)


def test_smart_orientation_not_transposed():
    # Landscape in, landscape out — catches any width/height swap regression.
    plan = smart(1920, 1080)
    assert plan.width > plan.height


def test_smart_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        smart(2010, 10)


def test_smart_rejects_unsatisfiable_max_pixels():
    # Elongated strip: aspect ratio passes the <=200 check, but the factor
    # floor cannot fit the pixel budget. Must raise, not silently overshoot.
    with pytest.raises(ValueError, match="cannot satisfy"):
        smart(40000, 200, factor=48, pixels=384)


def test_smart_rejects_unsatisfiable_min_pixels():
    # Tight budget + extreme shape floors below min_pixels.
    with pytest.raises(ValueError, match="cannot satisfy"):
        smart(20000, 10000, pixels=64)


def test_smart_min_pixels_defaults_to_four_patches():
    plan = smart(32, 32)
    assert plan.width * plan.height >= 4 * 32 * 32
    assert plan == smart(32, 32, min_pixels=4 * 32 * 32)


def test_smart_requires_factor():
    with pytest.raises(ValueError, match="requires factor"):
        compute_resize(
            width=1920,
            height=1080,
            name="smart_resize",
            pixels=768,
            interpolation="bicubic",
        )


def test_smart_rejects_min_pixels_over_budget():
    with pytest.raises(ValueError, match="min_pixels"):
        smart(1920, 1080, min_pixels=100000, pixels=64)
    # Implicit default min_pixels (4 * 48 * 48 = 9216) exceeds 64 * 64.
    with pytest.raises(ValueError, match="min_pixels"):
        smart(1920, 1080, factor=48, pixels=64)


@pytest.mark.parametrize(
    "kwargs",
    [{"factor": 0}, {"factor": -32}, {"factor": 32.5}, {"min_pixels": 0}],
)
def test_smart_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        smart(1920, 1080, **kwargs)


def test_square():
    plan = compute_resize(
        width=1920,
        height=1080,
        name="square_resize",
        pixels=384,
        interpolation="lanczos",
    )
    assert (plan.width, plan.height) == (384, 384)
    assert plan.interpolation == "lanczos"


def test_square_ignores_factor():
    # factor comes from the server resize spec, so forwarding the whole spec
    # must not blow up on a square model.
    plan = compute_resize(
        width=1920,
        height=1080,
        name="square_resize",
        pixels=384,
        factor=32,
        interpolation="bicubic",
    )
    assert (plan.width, plan.height) == (384, 384)


def test_square_rejects_min_pixels():
    with pytest.raises(ValueError, match="only applies to smart_resize"):
        compute_resize(
            width=1920,
            height=1080,
            name="square_resize",
            pixels=384,
            min_pixels=100,
            interpolation="bicubic",
        )


def test_positional_arguments_rejected():
    with pytest.raises(TypeError):
        compute_resize(
            1920, 1080, name="square_resize", pixels=384, interpolation="lanczos"
        )  # type: ignore[misc]


@pytest.mark.parametrize("width,height", [(0, 1080), (1920, 0), (0, 0), (-1, 100)])
def test_rejects_non_positive_dimensions(width, height):
    with pytest.raises(ValueError, match="height and width"):
        smart(width, height)


@pytest.mark.parametrize("pixels", [0, -384, 384.0, True, "384"])
def test_rejects_bad_pixels(pixels):
    # 384.0 is what a JSON-decoded UI option looks like; no imaging library
    # accepts float dimensions, so reject at the boundary, not downstream.
    with pytest.raises(ValueError, match="pixels must be a positive integer"):
        compute_resize(
            width=1920,
            height=1080,
            name="square_resize",
            pixels=pixels,
            interpolation="bicubic",
        )


def test_rejects_unknown_name():
    with pytest.raises(ValueError, match="name must be one of"):
        compute_resize(
            width=1920,
            height=1080,
            name="longest_edge",
            pixels=768,
            interpolation="bicubic",
        )


def test_rejects_unknown_interpolation():
    # bilinear is no longer a supported value.
    with pytest.raises(ValueError, match="interpolation must be one of"):
        smart(1920, 1080, interpolation="bilinear")
