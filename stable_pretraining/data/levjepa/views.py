"""The token loader's GPU stage: packed JPEGs of a slot -> every view's kept patches, normalized, on the GPU.

Same geometry, token draw and photometric ops as v2's ViewBuilder + PatchEmbed, with two differences:
  - decode uses per-MCU ("fast") chroma upsampling, which nvJPEG and the splice require (v2: libjpeg's fancy
    upsampling; about 1/255 mean difference)
  - contrast pivots on the mean of the view's kept patches over all its frames ("C2") instead of the per-frame mean
    of the full crop, which partial decoding never builds: v2's contrast plus a uniform shift delta * (1 - c)
"""

import torch

from stable_datasets.framestore.decoders import Decoder

from .recipe import V2
from .splice import PACK_COLS


def ring_slot(B, r, device="cuda", n=()):
    """Empty (global patches, global idx, local patches, local idx) for B clips; n=(k,): k of each, stacked."""
    T, P, V = r.frames, 3 * r.patch * r.patch, r.num_local * B
    return (
        torch.empty(*n, B, T * r.keep_global, P, device=device),
        torch.empty(*n, B, T * r.keep_global, dtype=torch.long, device=device),
        torch.empty(*n, V, T * r.keep_local, P, device=device),
        torch.empty(*n, V, T * r.keep_local, dtype=torch.long, device=device),
    )


class TokenViews:
    """The token loader's GPU stage: a slot's packed JPEGs -> kept patches.

    Slot -> (global patches (B, T*keep_global, 768), global idx, local patches (V*B, T*keep_local, 768), local
    idx), normalized like ViewBuilder.normalize and in PatchEmbed's (c p1 p2) order; idx index the flattened T * N
    grid.
    contrast: "view" (C2, the view's kept patches) or "frame" (each frame's kept patches of the view).
    decode: "partial" (packed MCUs) or "full" (ablation: whole frames, make_batch(decode="full")).
    fused: everything after decoding in Triton kernels (kernels/fused.py; C2 contrast and window squash only): one
    copies the decoded MCUs into the frame canvas, one builds every kept patch, one jitters the local views.
    """

    def __init__(
        self,
        B,
        r=V2,
        device="cuda",
        contrast="view",
        squash="windows",
        decode="partial",
        fused=False,
    ):
        self.B, self.r, self.dev, self.contrast, self.squash, self.decode = (
            B,
            r,
            device,
            contrast,
            squash,
            decode,
        )
        self.fused = fused
        assert not fused or (contrast == "view" and squash == "windows")
        self.mean = torch.tensor(r.mean, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(r.std, device=device).view(1, 1, 3, 1, 1)
        self.canvas = torch.empty(0, dtype=torch.uint8, device=device)
        self.decoder = Decoder("nvjpeg", device)
        self.grid = (
            0,
            0,
        )  # canvas size in MCUs; grows to the largest frame seen (e.g. 384 x 1211 is 24 x 76)

    def squashed(self, slots, s):
        """Squash slot s's frames whole, after decoding them into the canvas.

        Decode the packed JPEGs of slot s, scatter their MCUs into the frame canvas and squash each frame whole to
        squash x squash (antialiased bilinear, rounded to uint8 as v2's data.decode). Returns (B*T, 3, S, S) in [0, 1].
        """
        size = self.fill_canvas(slots, s)
        B, T, S = self.B, self.r.frames, self.r.squash
        out = torch.empty(B * T, 3, S, S, device=self.dev)
        clip_size = size.tolist()
        for hw in set(map(tuple, clip_size)):
            clips = [b for b in range(B) if tuple(clip_size[b]) == hw]
            fr = (
                (torch.tensor(clips)[:, None] * T + torch.arange(T)[None])
                .reshape(-1)
                .to(self.dev)
            )
            x = self.canvas[fr, :, : hw[0], : hw[1]].float()
            out[fr] = (
                torch.nn.functional.interpolate(
                    x, size=(S, S), mode="bilinear", antialias=True, align_corners=False
                )
                .round_()
                .clamp_(0, 255)
            )
        return out.div_(255)

    def fill_canvas(self, slots, s):
        """Decode slot s's packed JPEGs into the frame canvas.

        Decode slot s's packed JPEGs on the GPU and scatter their MCUs into the frame canvas; returns the
        (B, 2) stored frame sizes.
        """
        B, T, PS = self.B, self.r.frames, self.r.patch
        size = slots["size"][s]
        self._grow(B, size)
        jp = slots["jpeg"][s]
        if self.decode == "full":
            joff = slots["joff"][s].tolist()
            imgs = self.decoder([jp[joff[f] : joff[f + 1]] for f in range(B * T)])
            for b in range(B):  # the frames of a clip share one size
                h, w = imgs[b * T].shape[1:]
                self.canvas[b * T : (b + 1) * T, :, :h, :w] = torch.stack(
                    imgs[b * T : (b + 1) * T]
                )
            return size
        joff, moff = slots["joff"][s][: B + 1].tolist(), slots["moff"][s]
        imgs = self.decoder([jp[joff[b] : joff[b + 1]] for b in range(B)])
        if (
            self.fused
        ):  # one Triton program per MCU instead of cat + permute + index_put
            from .kernels import fused as tf

            m = slots["mcu"][s][: int(moff[-1])].to(self.dev, non_blocking=True)
            tf.scatter_mcus(
                imgs,
                slots["prow"][s].to(self.dev, non_blocking=True),
                m,
                self.canvas,
                T,
                PACK_COLS,
                slots["mdim"][s].to(self.dev, non_blocking=True),
            )
            return size
        m = slots["mcu"][s][: int(moff[-1])].to(self.dev, non_blocking=True).long()
        mdim = slots["mdim"][s].tolist()
        R, C = self.grid
        for mw, mh in set(
            map(tuple, mdim)
        ):  # one scatter per MCU size (16 x 16 for 4:2:0, 8 x 8 for 4:4:4, ...)
            clips = [b for b in range(B) if tuple(mdim[b]) == (mw, mh)]
            blocks = torch.cat(
                [
                    imgs[b]
                    .view(3, imgs[b].shape[1] // mh, mh, PACK_COLS, mw)
                    .permute(1, 3, 0, 2, 4)
                    .reshape(-1, 3, mh, mw)[: int(moff[b + 1] - moff[b])]
                    for b in clips
                ]
            )
            mm = torch.cat([m[int(moff[b]) : int(moff[b + 1])] for b in clips])
            cv = self.canvas.view(B * T, 3, R * PS // mh, mh, C * PS // mw, mw).permute(
                0, 2, 4, 1, 3, 5
            )
            cv[mm[:, 0], mm[:, 1], mm[:, 2]] = blocks
        return size

    def _grow(self, B, size):
        PS, T = self.r.patch, self.r.frames
        need = (-(-int(size[:, 0].max()) // PS), -(-int(size[:, 1].max()) // PS))
        if need[0] > self.grid[0] or need[1] > self.grid[1]:
            self.grid = (max(need[0], self.grid[0]), max(need[1], self.grid[1]))
            self.canvas = torch.empty(
                B * T,
                3,
                self.grid[0] * PS,
                self.grid[1] * PS,
                dtype=torch.uint8,
                device=self.dev,
            )

    def patches(self, frames, box, flip, idx, size, scale, clip_of_row):
        """Kept patches of views, as roi_align crops.

        Kept patches of views (rows) as raw [0, 1] crops (rows, S, 3, 16, 16), each sampled by roi_align with the
        sub-box of its view's box under that patch (the points ViewBuilder.crop samples), then flipped.
        """
        from torchvision.ops import roi_align

        PS, T, SQ = self.r.patch, self.r.frames, self.r.squash
        g = size // PS
        N = g * g
        idx = idx.long()
        t, n = idx // N, idx % N
        row, col = n // g, n % g
        col = torch.where(flip[:, None], g - 1 - col, col)
        x1, y1, x2, y2 = box[:, :1], box[:, 1:2], box[:, 2:3], box[:, 3:]
        bw, bh = (x2 - x1) / size * PS, (y2 - y1) / size * PS
        px, py = x1 + col * bw, y1 + row * bh
        frame = clip_of_row[:, None] * T + t
        sr = self.r.sampling_ratio(scale, size)
        if (
            frames is None
        ):  # squash only a window around each patch (kernels/squash.py), straight from the canvas
            from .kernels import squash as tq

            px, py, frame = px.reshape(-1), py.reshape(-1), frame.reshape(-1)
            bw, bh = (
                bw.expand(-1, idx.shape[1]).reshape(-1),
                bh.expand(-1, idx.shape[1]).reshape(-1),
            )
            # window placement keeps roi_align's frame-border handling: windows are clamped to the frame, and away
            # from its edges they extend past every bilinear tap
            r0 = ((py - 0.5).floor() - 1).clamp(0, SQ - tq.K).long()
            c0 = ((px - 0.5).floor() - 1).clamp(0, SQ - tq.K).long()
            win = tq.windows(self.canvas, self.hw_frames, frame, r0, c0, SQ)
            n = torch.arange(len(px), device=px.device, dtype=px.dtype)
            rois = torch.stack(
                [n, px - c0, py - r0, px + bw - c0, py + bh - r0], dim=-1
            )
            out = roi_align(
                win,
                rois,
                output_size=PS,
                spatial_scale=1.0,
                sampling_ratio=sr,
                aligned=True,
            ).view(len(box), -1, 3, PS, PS)
        else:
            rois = torch.stack([frame.float(), px, py, px + bw, py + bh], dim=-1).view(
                -1, 5
            )
            out = roi_align(
                frames,
                rois,
                output_size=PS,
                spatial_scale=1.0,
                sampling_ratio=sr,
                aligned=True,
            ).view(len(box), -1, 3, PS, PS)
        return torch.where(flip[:, None, None, None, None], out.flip(-1), out)

    @staticmethod
    def luminance(x):
        return 0.299 * x[:, :, 0:1] + 0.587 * x[:, :, 1:2] + 0.114 * x[:, :, 2:3]

    def jitter(self, x, jit, idx, size):
        """ViewBuilder.jitter on kept patches x (rows, S, 3, 16, 16); contrast pivots on the kept patches' mean."""
        bri, con, sat, gray = (jit[:, j].view(-1, 1, 1, 1, 1) for j in range(4))
        x = (x * bri).clamp_(0, 1)
        if self.contrast == "view":
            m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        else:  # per frame of the view, over its kept patches (frames with none keep their pixels' own mean)
            T = self.r.frames
            N = (size // self.r.patch) ** 2
            t = idx.long() // N
            pm = x.mean(dim=(2, 3, 4))  # (rows, S)
            tot = torch.zeros(x.shape[0], T, device=x.device).scatter_add_(1, t, pm)
            cnt = torch.zeros(x.shape[0], T, device=x.device).scatter_add_(
                1, t, torch.ones_like(pm)
            )
            m = (tot / cnt.clamp(min=1)).gather(1, t)[..., None, None, None]
        x = ((x - m) * con + m).clamp_(0, 1)
        lum = self.luminance(x)
        x = ((x - lum) * sat + lum).clamp_(0, 1)
        return torch.where(gray.bool(), self.luminance(x).expand_as(x), x)

    def fused_call(self, slots, s, out=None):
        """(global patches, idx, local patches, idx) via kernels/fused.py, into out (a ring slot) when given."""
        from .kernels import fused as tf

        B, dev, r = self.B, self.dev, self.r
        T = r.frames
        g = {
            k: slots[k][s].to(dev, non_blocking=True)
            for k in (
                "gbox",
                "gflip",
                "gidx",
                "lbox",
                "lflip",
                "lidx",
                "ljit",
                "gorder",
                "lorder",
            )
        }
        if out is None:
            out = ring_slot(B, r, dev)
        size = self.fill_canvas(slots, s).to(dev, non_blocking=True)
        hw = size.repeat_interleave(T, dim=0).int()
        tf.kept_patches(
            self.canvas,
            hw,
            B,
            T,
            g["gbox"],
            g["gflip"],
            g["gidx"],
            g["gorder"],
            slots["gcount"][s],
            r.global_size,
            r.sampling_ratio(r.global_scale, r.global_size),
            False,
            out[0],
            squash=r.squash,
            mean=r.mean,
            std=r.std,
        )
        tf.kept_patches(
            self.canvas,
            hw,
            B,
            T,
            g["lbox"],
            g["lflip"],
            g["lidx"],
            g["lorder"],
            slots["lcount"][s],
            r.local_size,
            r.sampling_ratio(r.local_scale, r.local_size),
            True,
            out[2],
            bri=g["ljit"][:, 0].contiguous(),
            jit=g["ljit"],
            squash=r.squash,
            mean=r.mean,
            std=r.std,
        )
        out[1].copy_(g["gidx"])
        out[3].copy_(g["lidx"])
        return out

    def __call__(self, slots, s, frames=None):
        if self.fused and frames is None:
            return self.fused_call(slots, s)
        B, dev, r = self.B, self.dev, self.r
        g = {
            k: slots[k][s].to(dev, non_blocking=True)
            for k in ("gbox", "gflip", "gidx", "lbox", "lflip", "lidx", "ljit")
        }
        if frames is None and self.squash == "full":
            frames = self.squashed(slots, s)
        elif frames is None:
            size = self.fill_canvas(slots, s).to(dev)
            self.hw_frames = size.repeat_interleave(r.frames, dim=0).int()
        clips = torch.arange(B, device=dev)
        gp = self.patches(
            frames,
            g["gbox"],
            g["gflip"],
            g["gidx"],
            r.global_size,
            r.global_scale,
            clips,
        )
        lp = self.patches(
            frames,
            g["lbox"],
            g["lflip"],
            g["lidx"],
            r.local_size,
            r.local_scale,
            clips.repeat(r.num_local),
        )
        lp = self.jitter(lp, g["ljit"], g["lidx"], r.local_size)
        gp, lp = (gp - self.mean) / self.std, (lp - self.mean) / self.std
        return gp.flatten(2), g["gidx"].long(), lp.flatten(2), g["lidx"].long()
