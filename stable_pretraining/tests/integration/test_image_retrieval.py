"""Integration tests for image retrieval functionality."""

import lightning as pl
import pytest
import torch
import torchmetrics
from transformers import AutoModel

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestImageRetrievalIntegration:
    """Integration tests for image retrieval with actual models and data."""

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_imgret_full_pipeline(self):
        """Test full image retrieval pipeline with DINO model."""
        # Load pre-trained DINO model
        backbone = AutoModel.from_pretrained("facebook/dino-vits16")

        # Define transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create train dataloader with small subset
        train = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="train[:128]",
                transform=train_transform,
            ),
            batch_size=128,
            shuffle=True,
            num_workers=10,
            drop_last=True,
        )

        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create image retrieval dataset
        imgret_ds = spt.data.HFDataset(
            path="randall-lab/revisitop",
            name="roxford5k",
            split="qimlist+imlist",
            trust_remote_code=True,
            transform=val_transform,
        )

        # Map query identification
        imgret_ds.dataset = imgret_ds.dataset.map(
            lambda example: {"is_query": example["query_id"] >= 0}
        )

        val = torch.utils.data.DataLoader(
            dataset=imgret_ds,
            batch_size=1,
            shuffle=False,
            num_workers=10,
        )

        # Create data module
        data = spt.data.DataModule(train=train, val=val)

        # Define forward function for feature extraction
        def forward(self, batch, stage):
            with torch.inference_mode():
                x = batch["image"]
                cls_embed = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
                batch["embedding"] = cls_embed
            return batch

        # Create module with eval-only backbone
        module = spt.Module(
            backbone=spt.backbone.EvalOnly(backbone), forward=forward, optim=None
        )

        # Create image retrieval callback
        img_ret = spt.callbacks.ImageRetrieval(
            module,
            "img_ret",
            input="embedding",
            query_col="is_query",
            retrieval_col=["easy", "hard"],
            features_dim=384,
            metrics={
                "mAP": torchmetrics.RetrievalMAP(),
                "R@1": torchmetrics.RetrievalRecall(top_k=1),
                "R@5": torchmetrics.RetrievalRecall(top_k=5),
                "R@10": torchmetrics.RetrievalRecall(top_k=10),
            },
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=0,
            num_sanity_val_steps=0,
            callbacks=[img_ret],
            precision="16",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training and validation
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        manager.validate()

    @pytest.mark.v1
    @pytest.mark.download
    def test_retrieval_dataset_loading(self):
        """Test loading and processing of retrieval datasets."""
        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(),
        )

        # Load retrieval dataset
        imgret_ds = spt.data.HFDataset(
            path="randall-lab/revisitop",
            name="roxford5k",
            split="qimlist+imlist",
            trust_remote_code=True,
            transform=val_transform,
        )

        # Map query identification
        imgret_ds.dataset = imgret_ds.dataset.map(
            lambda example: {"is_query": example["query_id"] >= 0}
        )

        # Test dataset properties
        sample = imgret_ds[0]
        assert "image" in sample
        assert "is_query" in sample
        assert isinstance(sample["is_query"], bool)

    @pytest.mark.gpu
    def test_feature_extraction_with_dino(self):
        """Test feature extraction using DINO model."""
        from transformers import AutoModel

        # Load model
        backbone = AutoModel.from_pretrained("facebook/dino-vits16")

        # Create dummy batch
        batch = {"image": torch.randn(2, 3, 224, 224)}

        # Extract features
        with torch.inference_mode():
            output = backbone(pixel_values=batch["image"])
            cls_embed = output.last_hidden_state[:, 0, :]

        # Verify output shape
        assert cls_embed.shape == (2, 384)  # DINO ViT-S/16 has 384-dim features

    @pytest.mark.v1
    def test_retrieval_metrics_computation(self):
        """Test retrieval metrics computation."""
        # Create dummy embeddings and labels
        embeddings = torch.randn(100, 384)
        is_query = torch.tensor([True] * 10 + [False] * 90)
        relevance = torch.randint(0, 2, (10, 90))  # Binary relevance

        # Initialize metrics
        metrics = {
            "mAP": torchmetrics.RetrievalMAP(),
            "R@1": torchmetrics.RetrievalRecall(top_k=1),
            "R@5": torchmetrics.RetrievalRecall(top_k=5),
            "R@10": torchmetrics.RetrievalRecall(top_k=10),
        }

        # Compute pairwise distances
        query_embeds = embeddings[is_query]
        gallery_embeds = embeddings[~is_query]
        distances = torch.cdist(query_embeds, gallery_embeds)

        # Convert distances to similarities (negative distance)
        similarities = -distances

        # Update metrics (for each query)
        for i in range(similarities.shape[0]):
            preds = similarities[i]
            target = relevance[i].bool()

            for metric in metrics.values():
                metric.update(preds, target)

        # Compute final metrics
        results = {name: metric.compute() for name, metric in metrics.items()}

        # Verify results are computed
        assert all(isinstance(v, torch.Tensor) for v in results.values())
        assert all(v.numel() == 1 for v in results.values())  # Single value per metric
