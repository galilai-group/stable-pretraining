"""Small frame stores for the tests.

Episodes of synthetic JPEG frames in a lance dataset, as a frame store holds them (episode_idx, frame, h, w), over
several fragments.
"""

import io

import numpy as np


def frame(h, w, seed, subsampling=2, quality=90):
    """A JPEG of smooth colour structure plus noise (so every MCU has AC energy)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = (
        np.stack(
            [np.sin(x / (5 + 3 * c) + seed) * np.cos(y / 7) * 90 + 120 + rng.random((h, w)) * 40 for c in range(3)], -1
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=quality, subsampling=subsampling)
    return b.getvalue()


def make_store(path, episodes=6, frames=40, size=(72, 104), subsampling=2, rows_per_fragment=64):
    """A store of `episodes` episodes of `frames` frames each; returns the frames' bytes in row order."""
    import lance
    import pyarrow as pa

    h, w = size
    blobs = [frame(h, w, 100 * e + t, subsampling) for e in range(episodes) for t in range(frames)]
    table = pa.table(
        {
            "episode_idx": pa.array(np.repeat(np.arange(episodes), frames), pa.int64()),
            "frame": pa.array(blobs, pa.large_binary()),
            "h": pa.array(np.full(len(blobs), h), pa.int32()),
            "w": pa.array(np.full(len(blobs), w), pa.int32()),
        }
    )
    lance.write_dataset(table, path, max_rows_per_file=rows_per_fragment)
    return blobs
