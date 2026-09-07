"""Image-resize planning helpers for VLM inference.

This module only computes target dimensions (pure math, no image
dependencies). The mode (``name``), ``factor``, and ``interpolation`` come
from the model server's resize spec; ``pixels`` comes from the UI resize
option. The actual resize is done by the caller with whatever imaging
library its pipeline already uses, applying ``plan.interpolation``:

    plan = compute_resize(
        width=w, height=h, name="smart_resize", factor=32, pixels=768,
        interpolation="bicubic",
    )

    # PIL client
    PIL_INTERPOLATION = {
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    out = img.resize((plan.width, plan.height), PIL_INTERPOLATION[plan.interpolation])

    # OpenCV client
    CV2_INTERPOLATION = {
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    out = cv2.resize(
        arr, (plan.width, plan.height), interpolation=CV2_INTERPOLATION[plan.interpolation]
    )
"""

import math
from typing import Literal, NamedTuple, get_args

_MAX_ASPECT_RATIO = 200

ResizeMode = Literal["smart_resize", "square_resize"]
Interpolation = Literal["bicubic", "lanczos"]


class ResizePlan(NamedTuple):
    """Target dimensions and interpolation method for a client-side resize."""

    width: int
    height: int
    interpolation: Interpolation


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
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
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
    *,
    width: int,
    height: int,
    name: ResizeMode,
    pixels: int,
    interpolation: Interpolation,
    factor: int | None = None,
    min_pixels: int | None = None,
) -> ResizePlan:
    """Plan a VLM-input resize from the model server's resize spec.

    All arguments are keyword-only — a transposed width/height cannot happen
    silently. ``name``, ``factor``, and ``interpolation`` mirror the model
    server's resize spec fields; ``pixels`` is the UI resize option value.

    - ``name="smart_resize"`` — dimensions divisible by ``factor``, total
      pixel count within [``min_pixels``, ``pixels * pixels``], aspect ratio
      kept. ``factor`` is required (the model's patch size, e.g. 32 or 48);
      ``min_pixels`` defaults to ``4 * factor * factor``.
    - ``name="square_resize"`` — exact ``pixels`` x ``pixels``; aspect ratio
      not preserved. ``factor`` has no meaning and is ignored, so a whole
      server resize spec can be forwarded as-is.

    Args:
        width: Source image width in pixels.
        height: Source image height in pixels.
        name: Resize mode from the model server: "smart_resize" or
            "square_resize".
        pixels: UI pixel option — one number, but its meaning depends on
            ``name``:

            - ``"smart_resize"``: an area budget, not a side length. The
              output area is capped at ``pixels * pixels`` while keeping the
              source aspect ratio, so neither output side is generally equal
              to ``pixels`` — e.g. ``pixels=768`` on a 1920x1080 frame gives
              1024x576 (= 589,824 px, the same area as 768x768).
            - ``"square_resize"``: the exact side length. The output is
              always ``pixels`` x ``pixels`` — e.g. ``pixels=384`` gives
              384x384 regardless of the source shape.
        interpolation: Interpolation the caller should resize with,
            from the model server: "bicubic" or "lanczos". Passed through
            into the returned plan.
        factor: Smart resize only: round dimensions to multiples of this.
        min_pixels: Smart resize only: lower bound on output pixel count.

    Returns:
        ResizePlan(width, height, interpolation) — note width-first,
        matching the (width, height) order PIL and OpenCV resize calls expect.

    Raises:
        ValueError: If height/width are not positive, ``pixels`` or ``factor``
            is not a positive integer, ``name`` or ``interpolation`` is not a
            supported value, ``factor`` is missing (smart), ``min_pixels`` is
            given for square resize or exceeds the pixel budget, or the
            constraints cannot be satisfied (extreme aspect ratio).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {width}x{height}")
    if name not in get_args(ResizeMode):
        raise ValueError(f"name must be one of {get_args(ResizeMode)}, got {name!r}")
    if interpolation not in get_args(Interpolation):
        raise ValueError(
            f"interpolation must be one of {get_args(Interpolation)}, "
            f"got {interpolation!r}"
        )
    if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels <= 0:
        raise ValueError(f"pixels must be a positive integer, got {pixels!r}")

    if name == "smart_resize":
        if factor is None:
            raise ValueError("smart_resize requires factor")
        if isinstance(factor, bool) or not isinstance(factor, int) or factor <= 0:
            raise ValueError(f"factor must be a positive integer, got {factor!r}")
        max_pixels = pixels * pixels
        if min_pixels is None:
            min_pixels = 4 * factor * factor
        if min_pixels <= 0 or min_pixels > max_pixels:
            raise ValueError(
                f"min_pixels must be in (0, pixels * pixels], got "
                f"min_pixels={min_pixels}, pixels={pixels}"
            )
        h_bar, w_bar = _smart_resize(height, width, factor, min_pixels, max_pixels)
    else:
        # factor is part of the server resize spec, so a caller forwarding the
        # whole spec may pass it; it has no meaning here, so ignore it.
        if min_pixels is not None:
            raise ValueError("min_pixels only applies to smart_resize")
        w_bar = h_bar = pixels

    return ResizePlan(width=w_bar, height=h_bar, interpolation=interpolation)
