"""CrossMAE: Rethinking Patch Dependence for Masked Autoencoders.

CrossMAE keeps the MAE encoder path intact but replaces the joint
self-attention decoder with a cross-attention decoder. Mask tokens query the
visible encoder tokens, so reconstruction cost scales with the number of
predicted masked patches rather than the full patch sequence.

References:
    Fu et al. "Rethinking Patch Dependence for Masked Autoencoders." TMLR 2025.
    https://arxiv.org/abs/2401.14391
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import CrossAttention, MAEDecoder, patchify


@dataclass
class CrossMAEOutput(ModelOutput):
    """Structured output of the :class:`CrossMAE` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


class _CrossMAEDecoderBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int,
        mlp_ratio: float,
        self_attn: bool,
        drop: float,
        attn_drop: float,
    ):
        super().__init__()
        self.self_attn_enabled = self_attn
        if self_attn:
            from stable_pretraining.backbone import Attention

            self.norm0 = nn.LayerNorm(decoder_dim)
            self.self_attn = Attention(
                decoder_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.cross_attn = CrossAttention(
            decoder_dim,
            context_dim=encoder_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(decoder_dim)
        self.mlp = Mlp(
            in_features=decoder_dim,
            hidden_features=int(decoder_dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.self_attn_enabled:
            x = x + self.self_attn(self.norm0(x))
        x = x + self.cross_attn(self.norm1(x), context)
        return x + self.mlp(self.norm2(x))


class _FeatureMapMixer(nn.Module):
    def __init__(self, n_maps: int, decoder_depth: int):
        super().__init__()
        self.n_maps = n_maps
        self.linear = nn.Linear(n_maps, decoder_depth, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(self.n_maps))

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        return self.linear(torch.stack(feature_maps, dim=-1))


class CrossMAEDecoder(MAEDecoder):
    """MAE decoder variant that cross-attends mask queries to encoder tokens."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        output_dim: int,
        num_patches: int,
        grid_size: tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        self_attn: bool,
        weight_feature_maps: bool,
        num_feature_maps: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__(
            embed_dim=decoder_dim,
            decoder_embed_dim=decoder_dim,
            output_dim=output_dim,
            num_patches=num_patches,
            depth=0,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_embed_type="sincos_2d",
            grid_size=grid_size,
        )
        pos_embed = self.transformer.pos_embed.clone()
        del self.transformer

        self.weight_feature_maps = weight_feature_maps
        self.register_buffer("pos_embed", pos_embed)

        if weight_feature_maps:
            self.feature_mixer = _FeatureMapMixer(num_feature_maps, depth)
            self.context_norms = nn.ModuleList(
                [nn.LayerNorm(encoder_dim) for _ in range(depth)]
            )

        self.blocks = nn.ModuleList(
            [
                _CrossMAEDecoderBlock(
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=self_attn,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, output_dim)
        self.apply(self._init_weights)
        if weight_feature_maps:
            self.feature_mixer.reset_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        context: torch.Tensor | list[torch.Tensor],
        query_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, n_queries = query_idx.shape
        x = self.mask_token.expand(B, n_queries, -1)
        pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=query_idx.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1]),
        )
        x = x + pos

        if self.weight_feature_maps:
            context = self.feature_mixer(context)
            for i, block in enumerate(self.blocks):
                x = block(x, self.context_norms[i](context[..., i]))
        else:
            for block in self.blocks:
                x = block(x, context)

        return self.pred(self.norm(x))


class CrossMAE(Module):
    """CrossMAE masked autoencoding with a cross-attention decoder.

    :param encoder_name: timm ViT name or pre-built ViT-like ``nn.Module``.
    :param decoder_embed_dim: Decoder hidden dimension.
    :param decoder_depth: Number of cross-attention decoder blocks.
    :param decoder_num_heads: Decoder attention heads.
    :param mask_ratio: Fraction of patches hidden from the encoder.
    :param kept_mask_ratio: Fraction of all patches reconstructed by the decoder.
    :param norm_pix_loss: Normalize target pixels per patch.
    :param weight_feature_maps: Mix selected encoder feature maps per decoder layer.
    :param use_feature_maps: Encoder block indices to mix. ``(-1,)`` means all blocks.
    :param use_input: Include the encoder input tokens in the feature-map mixer.
    :param self_attn: Add self-attention among decoder mask queries before cross-attn.
    :param mlp_ratio: MLP expansion ratio in decoder blocks.
    :param image_size: Input image size used to configure patch positions.
    :param in_channels: Number of image channels.
    :param pretrained: Load pretrained timm encoder weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        decoder_embed_dim: int = 256,
        decoder_depth: int = 12,
        decoder_num_heads: int = 8,
        mask_ratio: float = 0.75,
        kept_mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        weight_feature_maps: bool = True,
        use_feature_maps: Sequence[int] = (-1,),
        use_input: bool = True,
        self_attn: bool = False,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        in_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        if not 0.0 < kept_mask_ratio <= mask_ratio < 1.0:
            raise ValueError(
                "kept_mask_ratio and mask_ratio must satisfy "
                f"0 < kept_mask_ratio <= mask_ratio < 1, got "
                f"{kept_mask_ratio=} and {mask_ratio=}"
            )

        if isinstance(encoder_name, str):
            import timm

            self.encoder = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
            )
        else:
            self.encoder = encoder_name

        patch_embed = self.encoder.patch_embed
        patch_size = patch_embed.patch_size
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        grid_size = patch_embed.grid_size
        grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        num_patches = grid_size[0] * grid_size[1]
        embed_dim = self.encoder.embed_dim
        blocks = self.encoder.blocks

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.kept_mask_ratio = kept_mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.image_size = image_size
        self.in_channels = in_channels
        self._has_cls = (
            hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None
        )
        self._use_feature_maps = self._resolve_feature_maps(
            use_feature_maps, len(blocks)
        )
        self.use_input = use_input
        num_feature_maps = len(self._use_feature_maps) + int(use_input)

        self.decoder = CrossMAEDecoder(
            encoder_dim=embed_dim,
            decoder_dim=decoder_embed_dim,
            output_dim=in_channels * patch_size * patch_size,
            num_patches=num_patches,
            grid_size=grid_size,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            self_attn=self_attn,
            weight_feature_maps=weight_feature_maps,
            num_feature_maps=num_feature_maps,
        )

    @staticmethod
    def _resolve_feature_maps(indices: Sequence[int], depth: int) -> list[int]:
        indices = list(indices)
        if indices == [-1]:
            return list(range(depth))
        return [idx if idx >= 0 else depth + idx for idx in indices]

    def _random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        n_visible = int(N * (1.0 - self.mask_ratio))
        n_skipped = int(N * (self.mask_ratio - self.kept_mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_visible]
        visible = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask[:, : n_visible + n_skipped] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible, mask

    def _patch_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.encoder.pos_embed
        return x + pos_embed[:, -x.shape[1] :, :]

    def _append_cls(self, x: torch.Tensor) -> torch.Tensor:
        if not self._has_cls:
            return x
        cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._has_cls:
            return tokens[:, 0]
        return tokens.mean(dim=1)

    def _encode(
        self,
        images: torch.Tensor,
        mask: bool,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, torch.Tensor]:
        x = self.encoder.patch_embed(images)
        x = self._patch_pos_embed(x)
        if mask:
            x, pred_mask = self._random_mask(x)
        else:
            pred_mask = torch.zeros(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )

        x = self._append_cls(x)
        x = self.encoder.pos_drop(x)

        feature_maps = []
        if self.use_input:
            feature_maps.append(x)

        for idx, block in enumerate(self.encoder.blocks):
            x = block(x)
            if idx in self._use_feature_maps:
                feature_maps.append(x)

        encoded = self.encoder.norm(x)
        context = feature_maps if self.decoder.weight_feature_maps and mask else encoded
        return context, self._pool(encoded), pred_mask

    @staticmethod
    def _mask_indices(mask: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(mask.int(), dim=1, stable=True)
        return order[:, -int(mask.sum(dim=1)[0].item()) :]

    def _targets(self, images: torch.Tensor, query_idx: torch.Tensor) -> torch.Tensor:
        targets = patchify(
            images,
            patch_size=(self.in_channels, self.patch_size, self.patch_size),
        )
        targets = torch.gather(
            targets,
            dim=1,
            index=query_idx.unsqueeze(-1).expand(-1, -1, targets.shape[-1]),
        )
        if self.norm_pix_loss:
            mean = targets.mean(dim=-1, keepdim=True)
            var = targets.var(dim=-1, keepdim=True)
            targets = (targets - mean) / (var + 1e-6).sqrt()
        return targets

    def forward(self, images: torch.Tensor) -> CrossMAEOutput:
        if not self.training:
            _, embedding, _ = self._encode(images, mask=False)
            return CrossMAEOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=embedding,
            )

        context, embedding, mask = self._encode(images, mask=True)
        query_idx = self._mask_indices(mask)
        predictions = self.decoder(context, query_idx)
        targets = self._targets(images.to(predictions.dtype), query_idx)
        loss = F.mse_loss(predictions, targets)

        return CrossMAEOutput(
            loss=loss,
            embedding=embedding.detach(),
            predictions=predictions,
            targets=targets.detach(),
            mask=mask,
        )
