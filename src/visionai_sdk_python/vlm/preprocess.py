"""Image-resize planning helpers for VLM inference.

This module only computes target dimensions (pure math, no image
dependencies). The actual resize is done by the caller with whatever
imaging library its pipeline already uses, applying ``plan.resampling``:

    plan = compute_resize(h, w, **RESIZE_OPTIONS["smart_768_p32"])

    # PIL client
    PIL_RESAMPLING = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    out = img.resize((plan.width, plan.height), PIL_RESAMPLING[plan.resampling])

    # OpenCV client
    CV2_RESAMPLING = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    out = cv2.resize(
        arr, (plan.width, plan.height), interpolation=CV2_RESAMPLING[plan.resampling]
    )
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, get_args

_MAX_ASPECT_RATIO = 200

Resampling = Literal["bilinear", "lanczos", "bicubic"]

# UI resize options -> compute_resize kwargs. Single source of truth shared
# by all services; do not copy this table into service code. Interim until
# the model server serves the resize spec per model, then this table goes
# away and its kwargs arrive from that API instead. Read-only on purpose:
# mutating an entry would silently repoint every caller in the process.
RESIZE_OPTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        name: MappingProxyType(kwargs)
        for name, kwargs in {
            "square_384": {"square": 384, "resampling": "bilinear"},
            "square_512": {"square": 512, "resampling": "bilinear"},
            "longest_384": {"longest_edge": 384, "resampling": "bilinear"},
            "longest_512": {"longest_edge": 512, "resampling": "bilinear"},
            "longest_768": {"longest_edge": 768, "resampling": "bilinear"},
            "smart_384_p32": {
                "factor": 32,
                "max_pixels": 384 * 384,
                "resampling": "bilinear",
            },
            "smart_384_p48": {
                "factor": 48,
                "max_pixels": 384 * 384,
                "resampling": "bicubic",
            },
            "smart_512_p32": {
                "factor": 32,
                "max_pixels": 512 * 512,
                "resampling": "bilinear",
            },
            "smart_576_p48": {
                "factor": 48,
                "max_pixels": 576 * 576,
                "resampling": "bicubic",
            },
            "smart_768_p32": {
                "factor": 32,
                "max_pixels": 768 * 768,
                "resampling": "bilinear",
            },
            "smart_768_p48": {
                "factor": 48,
                "max_pixels": 768 * 768,
                "resampling": "bicubic",
            },
        }.items()
    }
)


class ResizePlan(NamedTuple):
    """Target dimensions and resampling method for a client-side resize."""

    width: int
    height: int
    resampling: Resampling


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Compute target (height, width) — note height-first, unlike ResizePlan.

    1. Both dimensions are divisible by ``factor``.
    2. The total number of pixels is within [``min_pixels``, ``max_pixels``].
    3. The aspect ratio is maintained as closely as possible.

    Raises ValueError when the aspect ratio exceeds 200 or the constraints
    cannot all be satisfied. Argument validation is compute_resize's job.
    """
    if max(height, width) / min(height, width) > _MAX_ASPECT_RATIO:
        raise ValueError(
            "absolute aspect ratio must be smaller than "
            f"{_MAX_ASPECT_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    if not min_pixels <= h_bar * w_bar <= max_pixels:
        raise ValueError(
            f"cannot satisfy min_pixels={min_pixels}, max_pixels={max_pixels} for a "
            f"{width}x{height} image with factor={factor}"
        )
    return h_bar, w_bar


def compute_resize(
    height: int,
    width: int,
    *,
    factor: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    size: tuple[int, int] | None = None,
    square: int | None = None,
    longest_edge: int | None = None,
    resampling: Resampling = "bilinear",
) -> ResizePlan:
    """Plan a VLM-input resize. One function, four modes; pick exactly one:

    - ``factor=..., max_pixels=...`` — smart resize: dimensions divisible by
      ``factor``, pixel count within [``min_pixels``, ``max_pixels``], aspect
      ratio kept. ``min_pixels`` defaults to ``4 * factor * factor`` (the
      Qwen2-VL convention).
    - ``size=(width, height)`` — exact resize; aspect ratio not preserved.
    - ``square=n`` — resize to n x n; aspect ratio not preserved.
    - ``longest_edge=n`` — scale so the longer side becomes n, keeping the
      aspect ratio. Never upscales: an image already smaller keeps its
      original dimensions.

    Args:
        height: Source image height in pixels.
        width: Source image width in pixels.
        factor: Smart resize: round dimensions to multiples of this.
        min_pixels: Smart resize: lower bound on output pixel count.
        max_pixels: Smart resize: upper bound on output pixel count.
        size: Target (width, height) for exact resize.
        square: Target side length for square resize.
        longest_edge: Target length of the longer side.
        resampling: Interpolation the caller should resize with:
            "bilinear" (default), "lanczos", or "bicubic". Passed through
            into the returned plan.

    Returns:
        ResizePlan(width, height, resampling) — note width-first, matching
        the (width, height) order PIL and OpenCV resize calls expect.

    Raises:
        ValueError: If height/width are not positive, not exactly one mode
            is selected, a target value is not positive, smart resize is
            missing ``factor`` or ``max_pixels``, ``min_pixels`` exceeds
            ``max_pixels``, the constraints cannot be satisfied (extreme
            aspect ratio), or ``resampling`` is not a supported value.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {width}x{height}")
    if resampling not in get_args(Resampling):
        raise ValueError(
            f"resampling must be one of {get_args(Resampling)}, got {resampling!r}"
        )

    smart = any(p is not None for p in (factor, min_pixels, max_pixels))
    modes = [m for m in (size, square, longest_edge) if m is not None]
    if smart + len(modes) != 1:
        raise ValueError(
            "pass exactly one mode: factor/min_pixels/max_pixels (smart), "
            "size, square, or longest_edge"
        )

    if smart:
        if factor is None or max_pixels is None:
            raise ValueError("smart resize requires both factor and max_pixels")
        if factor <= 0 or max_pixels <= 0:
            raise ValueError(
                f"factor and max_pixels must be positive, got "
                f"factor={factor}, max_pixels={max_pixels}"
            )
        if min_pixels is None:
            min_pixels = 4 * factor * factor
        if min_pixels <= 0 or min_pixels > max_pixels:
            raise ValueError(
                f"min_pixels must be in (0, max_pixels], got "
                f"min_pixels={min_pixels}, max_pixels={max_pixels}"
            )
        h_bar, w_bar = _smart_resize(height, width, factor, min_pixels, max_pixels)
    elif size is not None:
        w_bar, h_bar = size
        if w_bar <= 0 or h_bar <= 0:
            raise ValueError(f"size must be positive, got {size}")
    elif square is not None:
        if square <= 0:
            raise ValueError(f"square must be positive, got {square}")
        w_bar = h_bar = square
    else:
        assert longest_edge is not None
        if longest_edge <= 0:
            raise ValueError(f"longest_edge must be positive, got {longest_edge}")
        scale = min(1.0, longest_edge / max(height, width))
        w_bar = max(1, round(width * scale))
        h_bar = max(1, round(height * scale))

    return ResizePlan(width=w_bar, height=h_bar, resampling=resampling)
