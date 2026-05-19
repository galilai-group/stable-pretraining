"""Unit tests for OnlineKNN, focused on the num_classes resolution path (#373)."""

import logging as stdlib_logging

import pytest
import torch
import torchmetrics

from stable_pretraining.callbacks.knn import OnlineKNN


def _make_knn(num_classes=None, k=3):
    """Minimal OnlineKNN factory for unit testing the prediction path."""
    return OnlineKNN(
        name="knn",
        input="emb",
        target="label",
        queue_length=100,
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=8,
        k=k,
        num_classes=num_classes,
    )


@pytest.mark.unit
class TestOnlineKNNNumClasses:
    """Regression tests for #373: predictions must match the requested width."""

    def test_explicit_num_classes_overrides_inference(self):
        """When ``num_classes`` is set, predictions are exactly that wide.

        The queue here contains labels in {0, 1, 2} only, so inference would
        produce width=3 and crash a metric configured for 10 classes. The
        fix should produce width=10.
        """
        knn = _make_knn(num_classes=10)

        # Deterministic features + only 3 distinct labels in the "queue".
        cached_features = torch.randn(30, 8)
        cached_labels = torch.cat(
            [torch.zeros(10), torch.ones(10), torch.full((10,), 2.0)]
        ).long()
        # Current batch also only sees those 3 classes.
        features = torch.randn(4, 8)
        current_targets = torch.tensor([0, 1, 2, 0])

        preds = knn._compute_knn_predictions(
            features, cached_features, cached_labels, current_targets
        )

        assert preds.shape == (4, 10), (
            f"expected (4, 10), got {tuple(preds.shape)} — explicit "
            f"num_classes was ignored"
        )
        # The unobserved classes (3..9) get zero weight.
        assert torch.all(preds[:, 3:] == 0)

    def test_inferred_num_classes_warns_once(self, caplog):
        """Without ``num_classes``, we infer and warn — once per callback."""
        knn = _make_knn(num_classes=None)

        cached_labels = torch.tensor([0, 1, 2, 0, 1, 2])
        cached_features = torch.randn(6, 8)
        features = torch.randn(2, 8)
        targets = torch.tensor([0, 1])

        with caplog.at_level(stdlib_logging.WARNING):
            preds_a = knn._compute_knn_predictions(
                features, cached_features, cached_labels, targets
            )
            preds_b = knn._compute_knn_predictions(
                features, cached_features, cached_labels, targets
            )

        # Inferred width = max label + 1 = 3
        assert preds_a.shape == (2, 3)
        assert preds_b.shape == (2, 3)
        # Latch worked — only one warning emitted across two calls.
        warn_msgs = [
            r.message for r in caplog.records if "inferring num_classes" in r.message
        ]
        # Loguru may bypass the standard logging handler in some configs; we
        # only require that the latch flipped (a behavioural guarantee).
        assert knn._warned_inferred_num_classes is True
        # If captured via stdlib handler, ensure no duplicate.
        if warn_msgs:
            assert len(warn_msgs) == 1

    def test_inference_includes_current_batch_labels(self):
        """Current-batch labels factor into the inferred width.

        If a class first appears in the current batch (not yet in the queue),
        the inferred width still covers it — important for the typical case
        where validation batches see new classes early in training.
        """
        knn = _make_knn(num_classes=None)

        # Queue has only classes {0, 1}.
        cached_features = torch.randn(20, 8)
        cached_labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        # But the current batch contains class 5 (not in queue).
        features = torch.randn(3, 8)
        current_targets = torch.tensor([0, 1, 5])

        preds = knn._compute_knn_predictions(
            features, cached_features, cached_labels, current_targets
        )
        # Width = max(cached.max=1, current.max=5) + 1 = 6
        assert preds.shape == (3, 6)

    def test_explicit_num_classes_too_small_raises(self):
        """Raise a clear error if ``num_classes`` is below observed labels.

        Better than silently producing garbage predictions for the
        out-of-range class.
        """
        knn = _make_knn(num_classes=3)  # claims 3 classes
        cached_features = torch.randn(10, 8)
        # But the queue contains label 5 — incompatible with num_classes=3.
        cached_labels = torch.tensor([0, 1, 2, 5, 0, 1, 2, 0, 1, 2])
        features = torch.randn(2, 8)
        targets = torch.tensor([0, 1])

        with pytest.raises(ValueError, match="observed label 5 >= num_classes"):
            knn._compute_knn_predictions(
                features, cached_features, cached_labels, targets
            )

    def test_negative_or_zero_num_classes_rejected_at_init(self):
        """``num_classes`` must be a positive int when provided."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            OnlineKNN(
                name="knn",
                input="emb",
                target="label",
                queue_length=10,
                metrics={"acc": torchmetrics.classification.MulticlassAccuracy(2)},
                num_classes=0,
            )
        with pytest.raises(ValueError, match="num_classes must be positive"):
            OnlineKNN(
                name="knn",
                input="emb",
                target="label",
                queue_length=10,
                metrics={"acc": torchmetrics.classification.MulticlassAccuracy(2)},
                num_classes=-1,
            )

    def test_explicit_num_classes_unblocks_metric_with_partial_queue(self):
        """End-to-end: explicit ``num_classes`` rescues the partial-queue case.

        The metric that previously crashed (because the queue hadn't seen
        every class) now works when ``num_classes`` is set. Using
        ``MulticlassAccuracy(num_classes=10)`` with a queue containing only
        3 classes — pre-fix this raised an IndexError-shape-mismatch.
        """
        knn = _make_knn(num_classes=10)

        cached_features = torch.randn(30, 8)
        cached_labels = torch.cat(
            [torch.zeros(10), torch.ones(10), torch.full((10,), 2.0)]
        ).long()
        features = torch.randn(4, 8)
        targets = torch.tensor([0, 1, 2, 0])

        preds = knn._compute_knn_predictions(
            features, cached_features, cached_labels, targets
        )

        # Feed predictions into a MulticlassAccuracy(10) metric — no crash.
        metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
        metric.update(preds, targets)
        value = metric.compute()
        # Value is a finite float scalar.
        assert torch.is_tensor(value) and value.ndim == 0
        assert torch.isfinite(value)


@pytest.mark.unit
class TestOnlineKNNPredictionCorrectness:
    """Ground-truth tests for the k-NN scoring math.

    These use small, well-separated synthetic vectors so the top-k neighbours
    and their weighted-vote outcome are known by construction and can be
    asserted exactly.
    """

    def _knn(self, k=3, temperature=0.07, num_classes=2):
        return OnlineKNN(
            name="knn",
            input="emb",
            target="label",
            queue_length=100,
            metrics={
                "acc": torchmetrics.classification.MulticlassAccuracy(num_classes)
            },
            input_dim=2,
            k=k,
            temperature=temperature,
            distance_metric="euclidean",
            num_classes=num_classes,
        )

    def test_two_well_separated_clusters_k3(self):
        """Two tight clusters far apart. Query near cluster 0 → class 0.

        Cluster 0 sits near origin, cluster 1 sits near (10, 10). The query
        (0.5, 0.5) has all three nearest neighbours in class 0, so
        ``argmax(predictions) == 0`` and ``predictions[1] == 0`` exactly.
        """
        knn = self._knn(k=3, num_classes=2)

        cluster_0 = torch.tensor(
            [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]], dtype=torch.float32
        )
        cluster_1 = torch.tensor(
            [[10.0, 10.0], [10.0, 10.1], [10.1, 10.0]], dtype=torch.float32
        )
        cached_features = torch.cat([cluster_0, cluster_1], dim=0)
        cached_labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        query = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        preds = knn._compute_knn_predictions(
            query, cached_features, cached_labels, current_targets=None
        )

        assert preds.shape == (1, 2)
        # Class-1 gets zero weight because all top-3 neighbours are class 0.
        assert preds[0, 1].item() == 0.0
        # Class-0 receives all the weight — strictly positive.
        assert preds[0, 0].item() > 0.0
        assert preds.argmax(dim=1).item() == 0

    def test_k1_returns_single_neighbour_class(self):
        """k=1: prediction is the inverse-distance weight at the neighbour's slot.

        Everything else is zero — useful for verifying the weighting math
        in isolation from the sum-of-weights aggregation.
        """
        knn = self._knn(k=1, temperature=0.07, num_classes=3)

        # Class 0 at origin, class 1 far right, class 2 far up
        cached_features = torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=torch.float32
        )
        cached_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        query = torch.tensor([[9.5, 0.0]], dtype=torch.float32)
        preds = knn._compute_knn_predictions(
            query, cached_features, cached_labels, current_targets=None
        )

        assert preds.shape == (1, 3)
        # Closest neighbour is class 1 at distance 0.5.
        assert preds.argmax(dim=1).item() == 1
        # Unselected classes are exactly zero.
        assert preds[0, 0].item() == 0.0
        assert preds[0, 2].item() == 0.0
        # The single weight equals 1 / (distance + temperature) = 1 / (0.5 + 0.07).
        expected_weight = 1.0 / (0.5 + 0.07)
        assert preds[0, 1].item() == pytest.approx(expected_weight, rel=1e-5)

    def test_mixed_topk_weighted_vote(self):
        """Mixed top-k: winner is the class with the higher inverse-distance sum.

        Not just the higher count. Setup: top-3 neighbours are 2× class 0 at distance 1.0 and 1× class
        1 at distance 0.5. Class 1 has the larger inverse-distance weight
        (1/0.57) compared to either single class-0 neighbour (1/1.07), but
        class 0 has TWO contributions:

            class_0_total = 2 / 1.07 ≈ 1.869
            class_1_total = 1 / 0.57 ≈ 1.754

        So class 0 should still win, but narrowly. This guards against a
        regression where the implementation reverts to plain-majority voting.
        """
        knn = self._knn(k=3, temperature=0.07, num_classes=2)

        # Two class-0 neighbours at L2 distance 1.0 from origin, one class-1
        # neighbour at L2 distance 0.5. Far-away decoys ensure top-3 is
        # exactly these three.
        cached_features = torch.tensor(
            [
                [1.0, 0.0],  # class 0, dist 1.0
                [-1.0, 0.0],  # class 0, dist 1.0
                [0.5, 0.0],  # class 1, dist 0.5  (closest)
                [100.0, 0.0],  # class 0, far away decoy
                [100.0, 1.0],  # class 1, far away decoy
            ],
            dtype=torch.float32,
        )
        cached_labels = torch.tensor([0, 0, 1, 0, 1], dtype=torch.long)

        query = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        preds = knn._compute_knn_predictions(
            query, cached_features, cached_labels, current_targets=None
        )

        expected_c0 = 2.0 / (1.0 + 0.07)
        expected_c1 = 1.0 / (0.5 + 0.07)

        assert preds.shape == (1, 2)
        assert preds[0, 0].item() == pytest.approx(expected_c0, rel=1e-5)
        assert preds[0, 1].item() == pytest.approx(expected_c1, rel=1e-5)
        # Inverse-distance weighting picks class 0 here despite class 1 being
        # the single closest neighbour.
        assert preds.argmax(dim=1).item() == 0
        # Sanity: the contest is close — the margin shouldn't accidentally
        # explode if the math regresses.
        assert preds[0, 0].item() - preds[0, 1].item() == pytest.approx(
            expected_c0 - expected_c1, rel=1e-5
        )

    def test_batch_of_queries_each_resolved_independently(self):
        """Each query in a batch is classified by its own nearest neighbours.

        A batch with queries near different clusters should produce
        per-query class assignments, not a single dominant prediction.
        """
        knn = self._knn(k=3, num_classes=2)

        cluster_0 = torch.tensor(
            [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]], dtype=torch.float32
        )
        cluster_1 = torch.tensor(
            [[10.0, 10.0], [10.0, 10.1], [10.1, 10.0]], dtype=torch.float32
        )
        cached_features = torch.cat([cluster_0, cluster_1], dim=0)
        cached_labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        # Three queries: near cluster 0, near cluster 1, between (still
        # closer to 0 by a hair).
        queries = torch.tensor(
            [[0.3, 0.3], [9.7, 9.7], [4.9, 4.9]], dtype=torch.float32
        )
        preds = knn._compute_knn_predictions(
            queries, cached_features, cached_labels, current_targets=None
        )

        assert preds.shape == (3, 2)
        assert preds.argmax(dim=1).tolist() == [0, 1, 0]
