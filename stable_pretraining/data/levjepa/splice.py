"""Splicing (CPU, numba): the MCUs under a batch's kept patches -> one packed baseline JPEG per clip.

A converted frame store (stable_datasets.framestore.convert) holds every frame with a restart marker after each MCU and the length of
every MCU's entropy segment (seg_len), so any subset of a frame's MCUs, each followed by a restart marker, is a valid
scan. All frames of a clip share one header (sampling and tables), so each clip's needed MCUs, from all its frames,
go into one JPEG PACK_COLS MCUs wide; mcu records where each one belongs (frame, MCU row, MCU column, slot in its
clip's JPEG), for the GPU stage to scatter them back into a frame canvas after decoding.
"""

import os

import numpy as np

try:
    import numba
except ImportError:
    numba = None

from .sampling import sample_params, token_footprints

PACK_COLS = 64  # MCUs per row of a packed JPEG (1024 px for 16 x 16 MCUs)
MCU_COLS = 4  # per packed MCU: frame, MCU row, MCU column (in its frame), slot in its clip's packed JPEG

if numba is not None:

    @numba.njit(cache=True, nogil=True)
    def _parse_one(data, i):
        """Returns sof, sos_end, mcu_w, mcu_h, rows, cols, ok for the JPEG starting at byte i."""
        i += 2
        s = -1
        ok = True
        while True:
            m = data[i + 1]
            L = (np.int64(data[i + 2]) << 8) | np.int64(data[i + 3])
            if m == 0xC0:
                s = i
            elif 0xC1 <= m <= 0xCF and m != 0xC4 and m != 0xC8 and m != 0xCC:
                ok = False
            if m == 0xDA:
                break
            i += 2 + L
        sos_end = i + 2 + L
        if s < 0:
            return 0, sos_end, 8, 8, 0, 0, False
        h = (np.int64(data[s + 5]) << 8) | np.int64(data[s + 6])
        w = (np.int64(data[s + 7]) << 8) | np.int64(data[s + 8])
        hv = np.int64(data[s + 11])
        mw = 8 * (hv >> 4)
        mh = 8 * (hv & 15)
        return s, sos_end, mw, mh, (h + mh - 1) // mh, (w + mw - 1) // mw, ok

    @numba.njit(cache=True, nogil=True)
    def _headers(bufs, fid, pos):
        F = len(pos)
        sof = np.empty(F, np.int64)
        sos_end = np.empty(F, np.int64)
        geo = np.empty((F, 4), np.int64)
        ok = np.ones(F, np.bool_)
        for f in range(F):
            a, b, mw, mh, r, c, good = _parse_one(bufs[fid[f]], pos[f])
            sof[f] = a
            sos_end[f] = b
            geo[f, 0] = mw
            geo[f, 1] = mh
            geo[f, 2] = r
            geo[f, 3] = c
            ok[f] = good and (mw == 8 or mw == 16) and (mh == 8 or mh == 16)
        return sof, sos_end, geo, ok

    @numba.njit(cache=True, nogil=True)
    def _pack(
        bufs,
        fid,
        pos,
        sof,
        sos_end,
        seg_len,
        seg_base,
        mframe,
        mid,
        geo,
        B,
        T,
        cols,
        out,
        mcu,
    ):
        """Pack each clip's needed MCUs into one baseline JPEG.

        mframe/mid: needed MCUs sorted by frame then id; frame f's bytes start at bufs[fid[f]][pos[f]]. Writes B
        packed JPEGs into out and, per packed MCU, (frame, mcu row, mcu col, slot k in its clip's packed JPEG) into
        mcu. Returns JPEG offsets (B + 1), MCU offsets (B + 1), rows per JPEG.
        """
        joff = np.zeros(B + 1, np.int64)
        moff = np.zeros(B + 1, np.int64)
        rows = np.zeros(B, np.int64)
        # per-frame segment start offsets (relative to the frame's SOS end)
        starts = np.empty(len(seg_len), np.int64)
        for f in range(len(pos)):
            p = 0
            for j in range(seg_base[f], seg_base[f + 1]):
                starts[j] = p
                p += np.int64(seg_len[j]) + 2
        q = 0
        o = 0
        for b in range(B):
            q0 = q
            while q < len(mframe) and mframe[q] < (b + 1) * T:
                q += 1
            n = q - q0
            moff[b + 1] = moff[b] + n
            r = (n + cols - 1) // cols
            rows[b] = r
            f0 = b * T
            d0 = bufs[fid[f0]]
            hl = sos_end[f0] - pos[f0]
            out[o : o + hl] = d0[pos[f0] : sos_end[f0]]
            s = o + sof[f0] - pos[f0] + 5
            h, w = r * geo[f0, 1], cols * geo[f0, 0]
            out[s] = h >> 8
            out[s + 1] = h & 255
            out[s + 2] = w >> 8
            out[s + 3] = w & 255
            o += hl
            total = r * cols
            for k in range(total):
                i = q0 + min(k, n - 1)  # pad the last row by repeating the last MCU
                f = mframe[i]
                d = bufs[fid[f]]
                j = seg_base[f] + mid[i]
                a = sos_end[f] + starts[j]
                L = np.int64(seg_len[j])
                out[o : o + L] = d[a : a + L]
                o += L
                out[o] = 0xFF
                out[o + 1] = 0xD9 if k == total - 1 else 0xD0 + (k & 7)
                o += 2
                if k < n:
                    m = moff[b] + k
                    mcu[m, 0] = f
                    mcu[m, 1] = mid[i] // geo[f, 3]
                    mcu[m, 2] = mid[i] % geo[f, 3]
                    mcu[m, 3] = k
            joff[b + 1] = o
        return joff, moff, rows


def needed_mcus(frame, xa, ya, xb, yb, geo, halo=1.0):
    """The MCUs under token boxes, sorted by frame then id, unique.

    Tokens (frame index + box in stored pixels) -> (mcu frame, mcu id) sorted by frame then id, unique.
    `halo` widens each box by the resampler's support (1px for bilinear roi_align).
    """
    mw, mh, rows, cols = (geo[frame, j] for j in range(4))
    c0 = np.clip(np.floor((xa - halo) / mw), 0, cols - 1).astype(np.int64)
    c1 = np.clip(np.floor((xb + halo) / mw), 0, cols - 1).astype(np.int64)
    r0 = np.clip(np.floor((ya - halo) / mh), 0, rows - 1).astype(np.int64)
    r1 = np.clip(np.floor((yb + halo) / mh), 0, rows - 1).astype(np.int64)
    nc, nr = c1 - c0 + 1, r1 - r0 + 1
    cells = nc * nr
    rep = np.repeat(np.arange(len(frame)), cells)
    local = np.arange(cells.sum()) - np.repeat(np.cumsum(cells) - cells, cells)
    ids = (r0[rep] + local // nc[rep]) * cols[rep] + c0[rep] + local % nc[rep]
    key = np.unique(frame[rep].astype(np.int64) << 32 | ids)
    return key >> 32, key & 0xFFFFFFFF


def make_batch(store, rng, B, out_jpeg, out_mcu, r, stride=2, decode="partial"):
    """Make one batch: view parameters and packed JPEGs.

    One batch of B clips (stable_datasets.framestore.FrameStore `store`, every `stride`-th row): view parameters (recipe r) and
    packed JPEGs. out_jpeg / out_mcu: preallocated buffers (shared memory). decode="full" (ablation): the clips'
    whole frames instead, back to back, with per-frame offsets in joff.
    """
    T = r.frames
    rows = store.windows(rng, B, frames=T, stride=stride)
    if decode == "full":
        joff, h, w = store.copy_frames(rows, out_jpeg)
        p = sample_params(rng, B, r)
        p.update(
            joff=joff, size=np.stack([h[::T], w[::T]], 1).astype(np.int32), rows=rows
        )
        return _with_order(p, B, r)
    if store.read == "ranges":
        return _with_order(
            _batch_ranges(store, rng, B, rows, out_jpeg, out_mcu, r), B, r
        )
    fid, pos, seg_len, nseg, h, w = store.take(rows)
    sof, sos_end, geo, ok = _headers(store.bufs, fid, pos)
    assert ok.all(), "frames must be baseline JPEGs with 8 or 16 pixel MCUs"
    assert np.array_equal(geo[:, :2], np.repeat(geo[::T, :2], T, axis=0)), (
        "a clip's frames differ in subsampling"
    )
    assert np.array_equal(nseg, geo[:, 2] * geo[:, 3]), (
        "segment lengths do not match the MCU grid"
    )
    seg_base = np.concatenate(([0], np.cumsum(nseg)))
    H, W = h[::T].astype(np.float64), w[::T].astype(np.float64)
    p = sample_params(rng, B, r)
    mframe, mid = needed_mcus(*token_footprints(p, B, H, W, r), geo, halo=1.0)
    joff, moff, prow = _pack(
        store.bufs,
        fid,
        pos,
        sof,
        sos_end,
        seg_len,
        seg_base,
        mframe,
        mid,
        geo,
        B,
        T,
        PACK_COLS,
        out_jpeg,
        out_mcu,
    )
    p.update(
        joff=joff,
        moff=moff,
        prow=prow,
        size=np.stack([H, W], 1).astype(np.int32),
        rows=rows,
        mdim=geo[::T, :2].astype(np.int32),
    )
    return _with_order(p, B, r)


def _batch_ranges(store, rng, B, rows, out_jpeg, out_mcu, r):
    """Make a batch from byte-range reads (read="ranges").

    make_batch reading only what the splice uses: the needed MCUs' bytes and each clip's first frame header.
    A frame's header length is its size less its scan (MCU segments, each followed by a 2-byte RST/EOI marker), and
    its MCU grid follows from (h, w) for 4:2:0, so nothing else of the frame is read (4:2:0 stores only).
    """
    T = r.frames
    c = store.take_meta(rows)
    seg_len, nseg, size = c["seg"], c["nseg"], c["size"]
    seg_base = np.concatenate(([0], np.cumsum(nseg)))
    h, w = c["h"].astype(np.int64), c["w"].astype(np.int64)
    geo = np.stack(
        [np.full_like(h, 16), np.full_like(h, 16), -(-h // 16), -(-w // 16)], 1
    )
    assert np.array_equal(nseg, geo[:, 2] * geo[:, 3]), (
        "segment lengths do not match the MCU grid"
    )
    H, W = h[::T].astype(np.float64), w[::T].astype(np.float64)
    p = sample_params(rng, B, r)
    mframe, mid = needed_mcus(*token_footprints(p, B, H, W, r), geo, halo=1.0)
    L = seg_len.astype(np.int64) + 2
    start = np.cumsum(L) - L
    start -= np.repeat(
        start[seg_base[:-1]], nseg
    )  # each segment's offset in its frame's scan
    hl = size - np.add.reduceat(L, seg_base[:-1])  # header length
    j = seg_base[mframe] + mid
    f0 = np.arange(B) * T
    fr = np.concatenate((f0, mframe))
    a = np.concatenate((np.zeros(B, np.int64), hl[mframe] + start[j]))
    b = np.concatenate((hl[f0], hl[mframe] + start[j] + seg_len[j]))
    o = np.lexsort((a, fr))
    fr, a, b = fr[o], a[o], b[o]
    new = np.ones(len(fr), bool)
    new[1:] = (fr[1:] != fr[:-1]) | (
        a[1:] - b[:-1] > int(os.environ.get("FRAMESTORE_READ_GAP", "0"))
    )
    k = np.flatnonzero(new)
    fr, a, b = fr[k], a[k], np.maximum.reduceat(b, k)
    store.last_ranges = (len(fr), int((b - a).sum()))
    fid, pos = store.fetch_ranges(c["fid"], c["pos"], size, fr, a, b)
    sof0, sos0, geo0, ok = _headers(store.bufs, fid[f0], pos[f0])
    assert (
        ok.all()
        and np.array_equal(sos0 - pos[f0], hl[f0])
        and np.array_equal(geo0, geo[f0])
    ), "unexpected header"
    sof = np.zeros(len(fid), np.int64)
    sof[f0] = sof0
    joff, moff, prow = _pack(
        store.bufs,
        fid,
        pos,
        sof,
        pos + hl,
        seg_len,
        seg_base,
        mframe,
        mid,
        geo,
        B,
        T,
        PACK_COLS,
        out_jpeg,
        out_mcu,
    )
    p.update(
        joff=joff,
        moff=moff,
        prow=prow,
        size=np.stack([H, W], 1).astype(np.int32),
        rows=rows,
        mdim=geo[::T, :2].astype(np.int32),
    )
    return p


def _with_order(p, B, r):
    """Patch launch order for the fused GPU stage: each view type's patches sorted by their clip's tile size."""
    from .kernels.order import patch_order

    H, W = p["size"][:, 0].astype(np.float64), p["size"][:, 1].astype(np.float64)
    p["gorder"], p["gcount"] = patch_order(
        H, W, B, 1, r.frames * r.keep_global, r.squash
    )
    p["lorder"], p["lcount"] = patch_order(
        H, W, B, r.num_local, r.frames * r.keep_local, r.squash
    )
    return p
