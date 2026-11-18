from unittest.mock import patch

import pytest
import torch
from torch import nn

from protopnet.train.metrics import InterpretableTrainingMetrics


class TestInterpretableTrainingMetrics:
    # Fixtures for InterpretableTrainingMetrics
    @pytest.fixture
    def protopnet_mock(self):
        class ProtoPNetMock:
            def __call__(self, x, return_prototype_layer_output_dict=False):
                return {"prototype_activations": torch.rand(len(x), 2, 1, 2, 2)}

            def get_prototype_complexity(self):
                return {
                    "n_unique_proto_parts": torch.tensor(8),
                    "n_unique_protos": torch.tensor(4),
                    "prototype_sparsity": torch.tensor(0.5),
                }

        return ProtoPNetMock()

    @pytest.fixture
    def training_metrics(self, protopnet_mock) -> InterpretableTrainingMetrics:
        return InterpretableTrainingMetrics(
            protopnet=protopnet_mock,
            num_classes=2,
            part_num=2,
            proto_per_class=1,
            img_size=224,
            half_size=36,
        )

    # Test that TrainingMetrics initializes correctly
    def test_TrainingMetrics_initialization(self, training_metrics):
        for metric in [
            "accuracy",
            "weighted_auroc",
            "class_aurocs",
            "conf_mat",
            "prototype_stability",
            "prototype_consistency",
            "prototype_sparsity",
            "n_unique_proto_parts",
            "n_unique_protos",
        ]:
            assert metric in training_metrics.metrics

    def test_compute_dict_after_project(self, training_metrics):
        # Calculate the metrics from project first
        training_metrics.prototypes_embedded_any = True

        forward_args = {
            "img": torch.rand(2, 3, 224, 224),
            "target": torch.tensor([1, 0]),
            "sample_parts_centroids": [[], []],
            "sample_bounding_box": torch.randint(0, 224, (2, 4)),
        }

        forward_output = {
            "logits": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "prototype_activations": torch.rand(2, 2, 1, 2, 2),
        }

        training_metrics.update_all(forward_args, forward_output, phase="project")

        with patch(
            "protopnet.metrics.InterpMetrics.proto2part_and_masks",
            return_value=(torch.ones(224, 224), torch.ones(224, 224)),
        ):
            result = training_metrics.compute_dict()

        expected_keys = {
            "accuracy",
            "pr",
            "roc",
            "weighted_auroc",
            "class_aurocs",
            "conf_mat",
            "prototype_sparsity",
            "n_unique_protos",
            "n_unique_proto_parts",
            "prototype_consistency",
            "prototype_stability",
            "prototype_score",
            "acc_proto_score",
        }
        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys

        for key in expected_keys:
            assert isinstance(result["prototype_consistency"], torch.Tensor), (
                key,
                result[key],
            )
            if key.startswith("n_"):
                assert result[key] // 1 == result[key], key

            elif key == "pr":
                precision, recall, _ = result[key]
                assert all(((x >= 0) & (x <= 1)).all() for x in precision)
                assert all(((x >= 0) & (x <= 1)).all() for x in recall)
                assert len(precision) == 2, key
                assert precision[0].shape == torch.Size([2]), key
                assert len(recall) == 2, key
                assert recall[0].shape == torch.Size([2]), key
            elif key == "roc":
                fpr, tpr, _ = result[key]
                assert all(((x >= 0) & (x <= 1)).all() for x in fpr)
                assert all(((x >= 0) & (x <= 1)).all() for x in tpr)
                assert len(fpr) == 2, key
                assert fpr[0].shape == torch.Size([3]), key
                assert len(tpr) == 2, key
                assert tpr[0].shape == torch.Size([3]), key
            elif key == "class_aurocs":
                assert (result[key] >= 0).all(), key
                assert (result[key] <= 1).all(), key
                assert result[key].shape == (2,), key
            elif key == "conf_mat":
                assert (result[key] >= 0).all(), key
                assert result[key].shape == (2, 2), key
                assert result[key].sum() == 2, key
            else:
                assert result[key] >= 0 and result[key] <= 1, key

        assert (
            result["prototype_score"]
            == (
                result["prototype_consistency"]
                + result["prototype_stability"]
                + result["prototype_sparsity"]
            )
            / 3
        )

        assert (
            result["acc_proto_score"] == result["accuracy"] * result["prototype_score"]
        )

        # conf_mat memory is shared between compute_dict() calls
        result["conf_mat"] = result["conf_mat"].clone()

        # update the metrics where only accuracy gets calculated
        # change the labels
        forward_args["target"] = torch.tensor([1, 1])
        training_metrics.update_all(forward_args, forward_output, phase="last_only")

        # no patch
        result_2 = training_metrics.compute_dict()

        for field in ["accuracy", "weighted_auroc", "acc_proto_score"]:
            assert (
                result_2[field] != result[field]
            ), f"{field} should change when acc is recalculated"
            del result_2[field]
            del result[field]

        # check pr and roc matrix separately because list of tensors
        for field in ["pr", "roc"]:
            val1_a, val2_a, _ = result[field]
            val1_b, val2_b, _ = result_2[field]
            assert not all(
                torch.equal(x, y) for x, y in zip(val1_a, val1_b)
            ), f"{field} should change when acc is recalculated"
            assert not all(
                torch.equal(x, y) for x, y in zip(val2_a, val2_b)
            ), f"{field} should change when acc is recalculated"
            del result_2[field]
            del result[field]

        # check class_aurocs and confusion matrix separately because not single-valued
        for field in ["class_aurocs", "conf_mat"]:
            assert not torch.equal(
                result_2[field], result[field]
            ), f"{field} should change when acc is recalculated"
            del result_2[field]
            del result[field]

        assert result_2 == result, "other results should be cached"

    @pytest.mark.parametrize(
        "spatial_perturbation",
        ["gaussian", "mask", "gaussian_mask"],
    )
    def test_spatial_alignment_generates_modified_activations(
        self,
        spatial_perturbation
    ):
        class SpatialProtoPNetMock(nn.Module):
            class PrototypeLayer:
                prototype_class_identity = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]]
                )

            prototype_layer = PrototypeLayer()

            def __init__(self):
                super().__init__()
                self.was_training_during_forward = None
                self.forward_inputs = []

            def __call__(self, x, return_prototype_layer_output_dict=False):
                self.was_training_during_forward = self.training
                self.forward_inputs.append(x.detach().clone())
                activations = x[:, :2, :2, :2]
                return {
                    "logits": torch.zeros(x.shape[0], 2),
                    "prototype_activations": activations,
                }

            def get_prototype_complexity(self):
                return {
                    "n_unique_proto_parts": torch.tensor(1),
                    "n_unique_protos": torch.tensor(1),
                    "prototype_sparsity": torch.tensor(1.0),
                }

        model = SpatialProtoPNetMock()
        metrics = InterpretableTrainingMetrics(
            protopnet=model,
            num_classes=2,
            proto_per_class=1,
            part_num=2,
            img_size=2,
            half_size=1,
            spatial_alignment=True,
            spatial_perturbation=spatial_perturbation,
        )
        forward_args = {
            "img": torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4),
            "target": torch.tensor([0]),
        }
        forward_outputs = {
            "logits": torch.tensor([[1.0, 0.0]]),
            "prototype_activations": model(forward_args["img"])[
                "prototype_activations"
            ],
        }

        metrics.update_all(forward_args, forward_outputs, phase="eval")
        result = metrics.compute_dict()

        assert {"spatial_pac", "spatial_prc", "spatial_plc"} <= result.keys()
        assert model.was_training_during_forward is False
        assert model.training is True
        assert len(model.forward_inputs) == 2
        modified_image = model.forward_inputs[-1]
        assert not torch.equal(modified_image, forward_args["img"])
        protected_boxes = metrics._get_protected_boxes(
            forward_args["img"],
            forward_outputs["prototype_activations"],
            forward_args["target"],
        )
        for top, left, bottom, right in protected_boxes:
            assert torch.equal(
                modified_image[:, :, top:bottom, left:right],
                forward_args["img"][:, :, top:bottom, left:right],
            )
        if spatial_perturbation in {"mask", "gaussian_mask"}:
            assert (modified_image == 0).any()
        if spatial_perturbation in {"gaussian", "gaussian_mask"}:
            assert not torch.equal(
                modified_image,
                torch.where(modified_image == 0, modified_image, forward_args["img"]),
            )

    # Currently only ProtoPNet is supported since ProtoPool is not implemented yet, if/when it is implemented the test needs to be updated
    # As well as the InterpretableTrainingMetrics class to support ProtoPool.
    def test_spatial_alignment_rejects_incompatible_models(self, protopnet_mock):
        with pytest.raises(TypeError, match="ProtoPNext/ProtoPool-compatible"):
            InterpretableTrainingMetrics(
                protopnet=protopnet_mock,
                num_classes=2,
                proto_per_class=1,
                part_num=2,
                img_size=224,
                half_size=36,
                spatial_alignment=True,
            )

    def test_compute_dict_updates_spatial_results_when_metrics_are_cached(self):
        class SpatialProtoPNetMock(nn.Module):
            class PrototypeLayer:
                prototype_class_identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

            prototype_layer = PrototypeLayer()

            def get_prototype_complexity(self):
                return {
                    "n_unique_proto_parts": torch.tensor(1),
                    "n_unique_protos": torch.tensor(1),
                    "prototype_sparsity": torch.tensor(1.0),
                }

        metrics = InterpretableTrainingMetrics(
            protopnet=SpatialProtoPNetMock(),
            num_classes=2,
            proto_per_class=1,
            part_num=2,
            img_size=224,
            half_size=36,
            spatial_alignment=True,
        )
        metrics.prototypes_embedded_any = True
        metrics.prototype_metrics_cached = True
        metrics.cached_results = {"prototype_score": torch.tensor(0.5)}

        expected = {
            "spatial_pac": torch.tensor(12.0),
            "spatial_prc": torch.tensor(3.0),
            "spatial_plc": torch.tensor(7.0),
        }
        with (
            patch.object(
                metrics.metrics["accuracy"].metric,
                "compute",
                return_value=torch.tensor(0.8),
            ),
            patch.object(
                metrics.metrics["pr"].metric,
                "compute",
                return_value=(torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5]))
            ),
            patch.object(
                metrics.metrics["roc"].metric,
                "compute",
                return_value=(torch.tensor([0.0]), torch.tensor([0.5]), torch.tensor([0.5]))
            ),
            patch.object(
                metrics.metrics["weighted_auroc"].metric,
                "compute",
                return_value=torch.tensor(0.8),
            ),
            patch.object(
                metrics.metrics["class_aurocs"].metric,
                "compute",
                return_value=torch.tensor([0.8, 0.8]),
            ),
            patch.object(
                metrics.metrics["conf_mat"].metric,
                "compute",
                return_value=torch.zeros((2, 2)),
            ),
            patch.object(
                metrics.metrics["spatial_pac"].metric,
                "compute",
                return_value=expected["spatial_pac"],
            ),
            patch.object(
                metrics.metrics["spatial_prc"].metric,
                "compute",
                return_value=expected["spatial_prc"],
            ),
            patch.object(
                metrics.metrics["spatial_plc"].metric,
                "compute",
                return_value=expected["spatial_plc"],
            ),
        ):
            result = metrics.compute_dict()

        assert all(result[name] == value for name, value in expected.items())
