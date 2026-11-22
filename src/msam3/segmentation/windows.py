"""FlurSeg window geometry, free of torch so that the UI can estimate from it."""

IMAGE_SIZE = 768
LONG_SIDE = {"fast": 3200, "balanced": 4000, "accurate": None}


def scaled_size(width: int, height: int, resolution: str) -> tuple[int, int]:
    """Limit the long side to the profile. An image is never enlarged."""
    limit, longest = LONG_SIDE[resolution], max(width, height)
    if limit is None or longest <= limit:
        return width, height
    return max(1, round(width * limit / longest)), max(1, round(height * limit / longest))


def window_starts(length: int, window: int, stride: int) -> list[int]:
    last = max(0, length - window)
    return sorted({*range(0, last + 1, stride), last})


def window_count(width: int, height: int, resolution: str, window: int = IMAGE_SIZE) -> int:
    scaled = scaled_size(width, height, resolution)
    columns, rows = (len(window_starts(side, window, window // 2)) for side in scaled)
    return columns * rows
