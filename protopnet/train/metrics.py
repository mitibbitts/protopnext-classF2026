import logging
from dataclasses import dataclass
from typing import List, Union

import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
)

from ..metrics import (
    PartConsistencyScore,
    PartStabilityScore,
    add_gaussian_noise,
    add_random_mask,
)
from ..prototypical_part_model import ProtoPNet
from ..utilities.trainer_utilities import predicated_extend

log = logging.getLogger(__name__)


@dataclass
class TrainingMetric:
    name: str
    # min, max
    metric: torchmetrics.Metric

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, metric={self.metric})"

    def __str__(self):
        return f"{self.name}: {self.metric.compute()}"


class SpatialMisalignmentScore(torchmetrics.Metric):
    """Measure prototype activation, rank, or location change after an attack (adding noise).

    The three modes correspond to the metrics used by the spatial alignment
    benchmark: PAC, PRC, and PLC respectively.
    Inputs are prototype activation maps before and after the spatial attack.
    """

    full_state_update = False

    def __init__(self, mode: str, img_size: int = 224):
        super().__init__()
        mode = mode.lower()
        if mode not in {"pac", "prc", "plc"}:
            raise ValueError(f"Unknown spatial misalignment mode: {mode}")
        self.mode = mode
        self.img_size = img_size
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        proto_acts: torch.Tensor,
        proto_acts_modified: torch.Tensor,
        targets: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        if proto_acts.shape != proto_acts_modified.shape:
            raise ValueError(
                "Original and modified activations must have the same shape."
            )
        if proto_acts.ndim != 4:
            raise ValueError(
                "Prototype activations must have shape "
                "[batch, prototypes, height, width]."
            )
        if targets.ndim != 1 or targets.shape[0] != proto_acts.shape[0]:
            raise ValueError("Targets must have one class label per sample.")

        class_identity = prototype_class_identity.to(proto_acts.device)
        if class_identity.ndim == 2:
            if class_identity.shape[0] != proto_acts.shape[1]:
                raise ValueError(
                    "Prototype class identity has an invalid prototype dimension."
                )
            prototype_classes = class_identity.argmax(dim=1)
        elif class_identity.ndim == 1 and class_identity.shape[0] == proto_acts.shape[1]:
            prototype_classes = class_identity
        else:
            raise ValueError(
                "Prototype class identity must have shape "
                "[prototypes, classes] or [prototypes]."
            )

        batch_scores = []
        for sample_idx, target in enumerate(targets):
            target_prototypes = torch.where(prototype_classes == target)[0]
            if target_prototypes.numel() == 0:
                raise ValueError(f"No prototypes found for target class {int(target)}.")

            before = proto_acts[sample_idx].flatten(start_dim=1).amax(dim=1)
            after = proto_acts_modified[sample_idx].flatten(start_dim=1).amax(dim=1)
            top_target_offset = before[target_prototypes].argmax()
            top_target = target_prototypes[top_target_offset]

            if self.mode == "pac":
                before_top = before[top_target]
                after_top = after[top_target]
                if before_top == 0:
                    score = torch.where(
                        after_top == 0, before_top, before_top.new_tensor(100.0)
                    )
                else:
                    score = (1 - after_top / before_top) * 100
            elif self.mode == "prc":
                non_target = prototype_classes != target
                score = (
                    (after[non_target] > after[top_target]).sum()
                    - (before[non_target] > before[top_target]).sum()
                ).float()
            else:
                before_map = torch.nn.functional.interpolate(
                    proto_acts[sample_idx, top_target].view(1, 1, *proto_acts.shape[-2:]),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).flatten()
                after_map = torch.nn.functional.interpolate(
                    proto_acts_modified[sample_idx, top_target].view(
                        1, 1, *proto_acts.shape[-2:]
                    ),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).flatten()
                before_region = before_map > torch.quantile(before_map, 0.9)
                after_region = after_map > torch.quantile(after_map, 0.9)
                union = (before_region | after_region).sum()
                intersection = (before_region & after_region).sum()
                score = torch.where(
                    union == 0,
                    union.new_tensor(0.0),
                    (1 - intersection.float() / union.float()) * 100,
                )

            batch_scores.append(score.float())

        scores = torch.stack(batch_scores)
        self.total += scores.sum()
        self.count += scores.numel()

    def compute(self):
        return self.total / self.count.clamp_min(1)


# TODO: Make all the metrics be calculated in the same way
class TrainingMetrics:
    def __init__(
        self,
        metrics: List[TrainingMetric] = [],
        device: Union[str, torch.device] = "cpu",
    ):
        self.metrics = {metric.name: metric for metric in metrics}

        for metric in self.metrics.values():
            metric.metric.to(device)

    def start_epoch(self, phase: str):
        self.reset()

    def end_epoch(self, phase: str):
        self.reset()

    def metric_names(self):
        return list(self.metrics.keys())

    def reset(self):
        for _, metric in self.metrics.items():
            metric.metric.reset()

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        return {
            metric_name: metric.metric.compute()
            for metric_name, metric in self.metrics.items()
        }

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        if len(self.metrics) > 0:
            raise NotImplementedError("This method must be implemented in a subclass.")

        # otherwise, do nothing, which is fine


class InterpretableTrainingMetrics(TrainingMetrics):
    """
    This is a temporary implementation of training metrics that lets us easily aggregate the interpretable
    metrics during an initial training run.
    """

    def __init__(
        self,
        protopnet: ProtoPNet,
        num_classes: int,
        proto_per_class: int,
        part_num: int,
        img_size: 224,
        half_size: 36,
        device: Union[str, torch.device] = "cpu",
        acc_only: bool = False,
        spatial_alignment: bool = False,
        spatial_noise_std: float = 0.2,
        spatial_noise_eps: float = 0.25,
        spatial_perturbation: str = "gaussian",
        spatial_mask_fraction: float = 0.25,
    ):
        """
        Args:
            protopnet (ProtoPNet): The ProtoPNet model.
            num_classes (int): The number of classes in the dataset.
            proto_per_class (int): The default number of prototypes per class.
            part_num (int): The number of parts in the dataset.
            img_size (int): The size of the input images.
            half_size (int): The size of the half of the input image. See: metrics.InterpretableMetrics
            device (Union[str, torch.device], optional): The device to run the metrics on. Defaults to "cpu".
            acc_only (bool, optional): If True, only compute accuracy. Defaults to False.
            spatial_alignment (bool, optional): If True, compute spatial alignment metrics. Defaults to False.
            spatial_noise_std (float, optional): The standard deviation of the spatial noise (For use in spatial alignment). Defaults to 0.2.
            spatial_noise_eps (float, optional): The epsilon value for the spatial noise (For use in spatial alignment). Defaults to 0.25.
            spatial_perturbation (str, optional): The type of spatial perturbation to use (For use in spatial alignment). Defaults to "gaussian".
            spatial_mask_fraction (float, optional): The fraction of the image to mask (For use in spatial alignment). Defaults to 0.25.
        """

        super().__init__(
            metrics=predicated_extend(
                not acc_only,
                [
                    TrainingMetric(
                        name="accuracy",
                        metric=torchmetrics.Accuracy(
                            num_classes=num_classes,
                            task="multiclass",
                        ),
                    ),
                    TrainingMetric(
                        name="pr",
                        metric=MulticlassPrecisionRecallCurve(
                            num_classes=num_classes,
                        ),
                    ),
                    TrainingMetric(
                        name="roc",
                        metric=MulticlassROC(
                            num_classes=num_classes,
                        ),
                    ),
                    TrainingMetric(
                        name="weighted_auroc",
                        metric=MulticlassAUROC(
                            num_classes=num_classes,
                            average="weighted",
                            thresholds=None,
                        ),
                    ),
                    TrainingMetric(
                        name="class_aurocs",
                        metric=MulticlassAUROC(
                            num_classes=num_classes,
                            average=None,
                            thresholds=None,
                        ),
                    ),
                    TrainingMetric(
                        name="conf_mat",
                        metric=MulticlassConfusionMatrix(
                            num_classes=num_classes,
                        ),
                    ),
                ],
                [
                    TrainingMetric(
                        name="prototype_stability",
                        metric=PartStabilityScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_consistency",
                        metric=PartConsistencyScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_sparsity",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_proto_parts",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_protos",
                        metric=torchmetrics.MeanMetric(),
                    ),
                ],
            ),
            device=device,
        )

        self.generator = torch.Generator(device=device)
        self.protopnet = protopnet

        self.prototypes_embedded_last = False
        self.prototype_metrics_cached = False
        self.prototypes_embedded_any = False
        # FIXME - this is hack
        self.acc_only = acc_only

        # Spatial Misalignment Metrics
        self.spatial_alignment = spatial_alignment
        if spatial_noise_std < 0:
            raise ValueError("spatial_noise_std must be non-negative due to use in gaussian approximation.")
        if spatial_noise_eps < 0:
            raise ValueError("spatial_noise_eps must be non-negative due to use in gaussian approximation.")
        self.spatial_noise_std = spatial_noise_std
        self.spatial_noise_eps = spatial_noise_eps
        if spatial_perturbation not in {"gaussian", "mask", "gaussian_mask"}:
            raise ValueError(
                "spatial_perturbation must be 'gaussian', 'mask', or 'gaussian_mask'."
            )
        if not 0 < spatial_mask_fraction <= 1:
            raise ValueError("spatial_mask_fraction must be greater than 0 and at most 1.")
        self.spatial_perturbation = spatial_perturbation
        self.spatial_mask_fraction = spatial_mask_fraction

        if spatial_alignment:
            self._get_prototype_class_identity()

        if spatial_alignment:
            for mode in ("pac", "prc", "plc"):
                self.metrics[f"spatial_{mode}"] = TrainingMetric(
                    name=f"spatial_{mode}",
                    metric=SpatialMisalignmentScore(mode=mode, img_size=img_size).to(
                        device
                    ),
                )

    def metric_names(self):
        raw_metric_names = super().metric_names()
        if self.acc_only:
            return raw_metric_names
        else:
            return raw_metric_names + [
                "prototype_score",
                "acc_proto_score",
            ]

    def start_epoch(self, phase: str):
        # This assumes that project only has eval once - if eval was run on train first, it would break this
        if self.protopnet.prototypes_embedded() and not self.prototypes_embedded_last:
            self.reset()
            self.prototype_metrics_cached = False
        else:
            self.metrics["accuracy"].metric.reset()
            self.metrics["pr"].metric.reset()
            self.metrics["roc"].metric.reset()
            self.metrics["weighted_auroc"].metric.reset()
            self.metrics["class_aurocs"].metric.reset()
            self.metrics["conf_mat"].metric.reset()
            if self.spatial_alignment:
                for mode in ("pac", "prc", "plc"):
                    self.metrics[f"spatial_{mode}"].metric.reset()

        if self.protopnet.prototypes_embedded():
            self.prototypes_embedded_any = True
        self.prototypes_embedded_last = self.protopnet.prototypes_embedded()

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        """
        Update all the metrics.
        """

        self.update_accuracy(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_pr(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_roc(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_weighted_auroc(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_class_aurocs(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_conf_mat(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )

        if not self.acc_only and "sample_bounding_box" in forward_args:
            self.update_stability(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_consistency(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_prototype_sparsity()

        if (
            self.spatial_alignment and "prototype_activations" in forward_outputs
        ):
            if "prototype_activations_modified" not in forward_outputs:
                if "img" not in forward_args:
                    raise ValueError(
                        "Spatial alignment metrics require 'img' in forward_args "
                        "when modified activations are not provided."
                    )
                forward_outputs = {
                    **forward_outputs,
                    "prototype_activations_modified": self._compute_modified_activations(
                        forward_args["img"],
                        forward_outputs["prototype_activations"],
                        forward_args["target"],
                    ),
                }
            self.update_spatial_alignment(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )

    def _compute_modified_activations(
        self,
        image: torch.Tensor,
        proto_acts: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a perturbed image through the model without changing its mode.
        
        Parameters:
        - image (torch.Tensor): The input image tensor.
        - proto_acts (torch.Tensor): The prototype activations from the original image.
        - targets (torch.Tensor): The target class labels for the input images.

        Returns:
        - torch.Tensor: The prototype activations from the perturbed image.
        """
        protected_boxes = self._get_protected_boxes(image, proto_acts, targets)
        modified_img = image
        if self.spatial_perturbation in {"gaussian", "gaussian_mask"}:
            modified_img = add_gaussian_noise(
                modified_img,
                self.generator,
                std=self.spatial_noise_std,
                eps=self.spatial_noise_eps,
                protected_boxes=protected_boxes,
            )
        if self.spatial_perturbation in {"mask", "gaussian_mask"}:
            modified_img = add_random_mask(
                modified_img,
                self.generator,
                fraction=self.spatial_mask_fraction,
                protected_boxes=protected_boxes,
            )
        was_training = self.protopnet.training
        try:
            self.protopnet.eval()
            with torch.no_grad():
                modified_output = self.protopnet(
                    modified_img,
                    return_prototype_layer_output_dict=True,
                )
            return modified_output["prototype_activations"].detach()
        finally:
            self.protopnet.train(was_training)

    def _get_protected_boxes(
        self,
        image: torch.Tensor,
        proto_acts: torch.Tensor,
        targets: torch.Tensor,
    ) -> list:
        """
        Map each sample's top target prototype location into an image box.
        
        Parameters:
        - image (torch.Tensor): The input image tensor.
        - proto_acts (torch.Tensor): The prototype activations from the original image.
        - targets (torch.Tensor): The target class labels for the input images.

        Returns:
        - list: A list of tuples representing the protected boxes in the format (top, left, bottom, right).
        """
        prototype_class_identity = self._get_prototype_class_identity().to(
            proto_acts.device
        )
        if prototype_class_identity.ndim == 2:
            prototype_classes = prototype_class_identity.argmax(dim=1)
        else:
            prototype_classes = prototype_class_identity

        _, _, activation_height, activation_width = proto_acts.shape
        image_height, image_width = image.shape[-2:]
        height_scale = image_height / activation_height
        width_scale = image_width / activation_width
        protected_boxes = []

        for sample_idx, target in enumerate(targets):
            target_prototypes = torch.where(prototype_classes == target)[0]
            if target_prototypes.numel() == 0:
                continue
            target_acts = proto_acts[sample_idx, target_prototypes]
            target_max_values = target_acts.flatten(start_dim=1).amax(dim=1)
            top_target = target_prototypes[target_max_values.argmax()]
            activation_map = proto_acts[sample_idx, top_target]
            location = torch.argmax(activation_map.flatten())
            activation_row = int(location // activation_width)
            activation_col = int(location % activation_width)

            center_row = round((activation_row + 0.5) * height_scale)
            center_col = round((activation_col + 0.5) * width_scale)
            radius_row = max(1, round(height_scale))
            radius_col = max(1, round(width_scale))
            top = max(0, center_row - radius_row)
            left = max(0, center_col - radius_col)
            bottom = min(image_height, center_row + radius_row)
            right = min(image_width, center_col + radius_col)
            protected_boxes.append((top, left, bottom, right))

        return protected_boxes

    def update_spatial_alignment(self, forward_args: dict, forward_outputs: dict):
        """Update PAC, PRC, and PLC from original and attacked activations."""
        prototype_class_identity = self._get_prototype_class_identity()

        for mode in ("pac", "prc", "plc"):
            self.metrics[f"spatial_{mode}"].metric.update(
                proto_acts=forward_outputs["prototype_activations"],
                proto_acts_modified=forward_outputs[
                    "prototype_activations_modified"
                ],
                targets=forward_args["target"],
                prototype_class_identity=prototype_class_identity,
            )

    def _get_prototype_class_identity(self) -> torch.Tensor:
        """Return class identity for a ProtoPNext/ProtoPool-compatible model."""
        prototype_class_identity = getattr(self.protopnet, "prototype_class_identity", None)
        if prototype_class_identity is None:
            prototype_layer = getattr(self.protopnet, "prototype_layer", None)
            prototype_class_identity = getattr(
                prototype_layer, "prototype_class_identity", None
            )
        if not isinstance(prototype_class_identity, torch.Tensor):
            raise TypeError(
                "Spatial alignment requires a ProtoPNext/ProtoPool-compatible "
                "model with a tensor prototype_class_identity."
            )
        if prototype_class_identity.ndim not in (1, 2):
            raise ValueError(
                "Spatial alignment requires prototype_class_identity with one "
                "or two dimensions."
            )
        return prototype_class_identity

    def update_accuracy(self, forward_args: dict, forward_outputs: dict):
        """
        Update the accuracy metric.
        """
        accuracy = self.metrics["accuracy"].metric
        accuracy.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_pr(self, forward_args: dict, forward_outputs: dict):
        """
        Update the precision-recall metric.
        """
        pr = self.metrics["pr"].metric
        pr.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_roc(self, forward_args: dict, forward_outputs: dict):
        """
        Update the roc metric.
        """
        roc = self.metrics["roc"].metric
        roc.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_weighted_auroc(self, forward_args: dict, forward_outputs: dict):
        """
        Update the weighted auroc metric.
        """
        weighted_auroc = self.metrics["weighted_auroc"].metric
        weighted_auroc.update(
            preds=forward_outputs["logits"], target=forward_args["target"]
        )

    def update_class_aurocs(self, forward_args: dict, forward_outputs: dict):
        """
        Update the class aurocs metric.
        """
        class_aurocs = self.metrics["class_aurocs"].metric
        class_aurocs.update(
            preds=forward_outputs["logits"], target=forward_args["target"]
        )

    def update_conf_mat(self, forward_args: dict, forward_outputs: dict):
        """
        Update the confusion matrix metric.
        """
        conf_mat = self.metrics["conf_mat"].metric
        conf_mat.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_stability(self, forward_args: dict, forward_outputs: dict):
        """
        Update the stability metric.
        """
        with torch.no_grad():
            # Compute the consistency metric (prototype_activations_noisy)
            proto_acts_noisy = self.protopnet(
                add_gaussian_noise(forward_args["img"], self.generator),
                return_prototype_layer_output_dict=True,
            )["prototype_activations"].detach()

        stability = self.metrics["prototype_stability"].metric
        stability.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            proto_acts_noisy=proto_acts_noisy,
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_consistency(self, forward_args: dict, forward_outputs: dict):
        """
        Update the consistency metric.
        """
        consistency = self.metrics["prototype_consistency"].metric

        consistency.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_prototype_sparsity(self):
        """
        Update the prototype sparsity metric.
        """
        prototype_complexity_stats = self.protopnet.get_prototype_complexity()

        self.metrics["prototype_sparsity"].metric.update(
            prototype_complexity_stats["prototype_sparsity"]
        )
        self.metrics["n_unique_protos"].metric.update(
            prototype_complexity_stats["n_unique_protos"]
        )
        self.metrics["n_unique_proto_parts"].metric.update(
            prototype_complexity_stats["n_unique_proto_parts"]
        )

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        spatial_results = {}
        if self.spatial_alignment:
            spatial_results = {
                metric_name: self.metrics[metric_name].metric.compute()
                for metric_name in ("spatial_pac", "spatial_prc", "spatial_plc")
            }

        if self.acc_only or not self.prototypes_embedded_any:
            return {
                "accuracy": self.metrics["accuracy"].metric.compute(),
                "pr": self.metrics["pr"].metric.compute(),
                "roc": self.metrics["roc"].metric.compute(),
                "weighted_auroc": self.metrics["weighted_auroc"].metric.compute(),
                "class_aurocs": self.metrics["class_aurocs"].metric.compute(),
                "conf_mat": self.metrics["conf_mat"].metric.compute(),
                **spatial_results,
            }

        if self.prototype_metrics_cached:
            log.debug("returning cached metrics")
            result_dict = self.cached_results
            result_dict["accuracy"] = self.metrics["accuracy"].metric.compute()
            result_dict["pr"] = self.metrics["pr"].metric.compute()
            result_dict["roc"] = self.metrics["roc"].metric.compute()
            result_dict["weighted_auroc"] = self.metrics[
                "weighted_auroc"
            ].metric.compute()
            result_dict["class_aurocs"] = self.metrics["class_aurocs"].metric.compute()
            result_dict["conf_mat"] = self.metrics["conf_mat"].metric.compute()
            result_dict.update(spatial_results)

        else:
            log.debug("calculating new metrics")
            result_dict = {
                metric_name: metric.metric.compute()
                for metric_name, metric in self.metrics.items()
            }

            result_dict["prototype_score"] = (
                min(result_dict["prototype_sparsity"], 1.0)
                + result_dict["prototype_stability"]
                + result_dict["prototype_consistency"]
            ) / 3

            self.prototype_metrics_cached = True
            self.cached_results = result_dict.copy()

        result_dict["acc_proto_score"] = (
            result_dict["prototype_score"] * result_dict["accuracy"]
        )

        return result_dict
