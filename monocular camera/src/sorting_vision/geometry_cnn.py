from __future__ import annotations

import json
import random
import csv
import shutil
import time
import ctypes
import importlib.util
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .geometry_models import GeometryCandidate, GeometryPrediction, GeometryShapeModel
from .geometry_rgb import (
    GeometryRGBModel,
    GeometrySample,
    load_geometry_samples,
    preprocess_geometry_object,
)


CNN_MODEL_VERSION = 1
DEFAULT_INPUT_SIZE = 192
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)
_TORCH_DLL_DIRECTORY = None
_TORCH_DLL_HANDLES: list[Any] = []


def _prepare_windows_torch_dlls() -> None:
    """Preload wheel DLLs when a non-ASCII project path breaks Windows lookup."""
    global _TORCH_DLL_DIRECTORY
    if os.name != "nt" or _TORCH_DLL_HANDLES:
        return
    specification = importlib.util.find_spec("torch")
    if specification is None or specification.origin is None:
        return
    library_dir = Path(specification.origin).parent / "lib"
    if not library_dir.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        _TORCH_DLL_DIRECTORY = os.add_dll_directory(str(library_dir.resolve()))
    priorities = {
        "libiomp5md.dll": 0,
        "c10.dll": 1,
        "torch_cpu.dll": 2,
        "torch.dll": 3,
        "torch_python.dll": 4,
    }
    libraries = sorted(
        library_dir.glob("*.dll"),
        key=lambda path: (priorities.get(path.name.lower(), 10), path.name.lower()),
    )
    for library in libraries:
        _TORCH_DLL_HANDLES.append(ctypes.WinDLL(str(library.resolve())))


def _optional_imports(training: bool = False):
    _prepare_windows_torch_dlls()
    try:
        import torch
        import torchvision
    except (ImportError, OSError) as error:
        extra = "cnn-train" if training else "cnn"
        raise RuntimeError(
            f"CNN dependencies are not installed; run pip install -e '.[{extra}]'"
        ) from error
    return torch, torchvision


def _prepare_object(
    image_bgr: np.ndarray, mask: np.ndarray | None, input_size: int
):
    prepared = None
    if mask is not None:
        prepared = preprocess_geometry_object(
            image_bgr, supplied_mask=mask, output_size=input_size
        )
    if prepared is None:
        prepared = preprocess_geometry_object(image_bgr, output_size=input_size)
    if prepared is None:
        return None, "object_not_found"
    if prepared.candidate_count != 1:
        return None, "multiple_objects"
    return prepared, "accepted"


def cnn_input_tensor(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    values = np.exp(values)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


class OpenVINOGeometryModel:
    backend = "openvino"

    def __init__(
        self,
        compiled_model,
        labels: list[str],
        class_names: dict[str, str],
        input_size: int = DEFAULT_INPUT_SIZE,
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.12,
        device: str = "CPU",
    ) -> None:
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("CNN metadata requires unique non-empty labels")
        self.compiled_model = compiled_model
        self.labels = list(labels)
        self.class_names = dict(class_names)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str = "CPU") -> "OpenVINOGeometryModel":
        directory = Path(path)
        metadata_path = directory / "metadata.json" if directory.is_dir() else directory.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"CNN metadata does not exist: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("model_version", -1)) != CNN_MODEL_VERSION:
            raise ValueError("unsupported geometry CNN model version")
        model_file = metadata_path.parent / metadata["model_file"]
        if not model_file.is_file():
            raise FileNotFoundError(f"OpenVINO model does not exist: {model_file}")
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "OpenVINO is not installed; run pip install -e '.[cnn]'"
            ) from error
        core = ov.Core()
        compiled = core.compile_model(model_file, device)
        return cls(
            compiled,
            metadata["labels"],
            metadata["class_names"],
            metadata.get("input_size", DEFAULT_INPUT_SIZE),
            metadata.get("confidence_threshold", 0.65),
            metadata.get("margin_threshold", 0.12),
            device,
        )

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        result = self.compiled_model([batch])
        if isinstance(result, Mapping):
            result = next(iter(result.values()))
        elif not isinstance(result, np.ndarray):
            result = np.asarray(result)
        return np.asarray(result, np.float32).reshape(len(batch), -1)

    def _prediction_from_probabilities(
        self, probabilities: np.ndarray, inference_ms: float
    ) -> GeometryPrediction:
        order = np.argsort(probabilities)[::-1]
        best_index = int(order[0])
        second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        best = float(probabilities[best_index])
        margin = best - second
        accepted = best >= self.confidence_threshold and margin >= self.margin_threshold
        label = self.labels[best_index] if accepted else "unknown"
        reason = "accepted" if accepted else (
            "confidence_rejected" if best < self.confidence_threshold else "margin_rejected"
        )
        candidates = tuple(
            GeometryCandidate(self.labels[int(index)], float(probabilities[index]))
            for index in order[: min(3, len(order))]
        )
        return GeometryPrediction(
            label,
            self.class_names.get(label, "未知形状"),
            best,
            accepted,
            self.backend,
            reason,
            candidates,
            inference_ms,
        )

    def predict_batch(
        self, items: Iterable[tuple[np.ndarray, np.ndarray | None]]
    ) -> list[GeometryPrediction]:
        values = list(items)
        outputs: list[GeometryPrediction | None] = [None] * len(values)
        tensors: list[np.ndarray] = []
        valid_indices: list[int] = []
        preprocessing_started = time.perf_counter()
        for index, (image, mask) in enumerate(values):
            prepared, reason = _prepare_object(image, mask, self.input_size)
            if prepared is None:
                outputs[index] = GeometryPrediction(
                    "unknown", "未知形状", 0.0, False, self.backend, reason
                )
                continue
            tensors.append(cnn_input_tensor(prepared.image_bgr))
            valid_indices.append(index)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        if tensors:
            batch = np.stack(tensors)
            started = time.perf_counter()
            logits = self._infer(batch)
            inference_ms = (time.perf_counter() - started) * 1000.0
            if logits.shape[1] != len(self.labels):
                raise ValueError(
                    f"CNN output has {logits.shape[1]} classes, expected {len(self.labels)}"
                )
            probabilities = _softmax(logits)
            per_item_ms = (inference_ms + preprocessing_ms) / len(tensors)
            for row, output_index in enumerate(valid_indices):
                outputs[output_index] = self._prediction_from_probabilities(
                    probabilities[row], per_item_ms
                )
        return [item for item in outputs if item is not None]

    def predict_geometry(
        self, image_bgr: np.ndarray, mask: np.ndarray | None = None
    ) -> GeometryPrediction:
        return self.predict_batch([(image_bgr, mask)])[0]

    def __call__(self, crop: np.ndarray, crop_mask: np.ndarray) -> tuple[str, float]:
        prediction = self.predict_geometry(crop, crop_mask)
        return prediction.label_id, prediction.confidence


def load_geometry_shape_model(
    backend: str, model_path: str | Path, device: str = "CPU"
) -> GeometryShapeModel:
    if backend == "opencv":
        return GeometryRGBModel.load(model_path)
    if backend == "openvino":
        return OpenVINOGeometryModel.load(model_path, device=device)
    raise ValueError(f"unsupported geometry backend: {backend}")


def _build_network(class_count: int, pretrained: bool):
    torch, torchvision = _optional_imports(training=True)
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, class_count)
    return model


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = image.shape[0]
    angle = float(rng.uniform(-30.0, 30.0))
    scale = float(rng.uniform(0.88, 1.12))
    tx, ty = rng.uniform(-0.07, 0.07, size=2) * size
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, matrix, (size, size), borderValue=(245, 245, 245))
    perspective = float(rng.uniform(-0.035, 0.035) * size)
    source = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    destination = source + np.float32(
        [[perspective, 0], [0, -perspective], [-perspective, 0], [0, perspective]]
    )
    result = cv2.warpPerspective(
        result,
        cv2.getPerspectiveTransform(source, destination),
        (size, size),
        borderValue=(245, 245, 245),
    )
    gain = float(rng.uniform(0.78, 1.22))
    bias = float(rng.uniform(-18.0, 18.0))
    result = np.clip(result.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    colour_gain = np.asarray(
        [rng.uniform(0.9, 1.1), rng.uniform(0.95, 1.05), rng.uniform(0.9, 1.1)],
        np.float32,
    )
    result = np.clip(result.astype(np.float32) * colour_gain, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        shadow = np.ones((size, size), np.float32)
        start = int(rng.uniform(0.15, 0.55) * size)
        shadow[:, start:] *= float(rng.uniform(0.65, 0.9))
        result = np.clip(result.astype(np.float32) * shadow[:, :, None], 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        result = cv2.GaussianBlur(result, (3, 3), float(rng.uniform(0.2, 1.2)))
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
    return result


class _GeometryTorchDataset:
    def __init__(self, samples: list[GeometrySample], labels: list[str], augment: bool, seed: int):
        torch, _ = _optional_imports(training=True)
        self.torch = torch
        self.samples = samples
        self.label_index = {label: index for index, label in enumerate(labels)}
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.images: list[np.ndarray] = []
        for sample in samples:
            prepared, reason = _prepare_object(
                sample.image_bgr, None, DEFAULT_INPUT_SIZE
            )
            if prepared is None:
                raise ValueError(f"cannot preprocess {sample.path}: {reason}")
            self.images.append(prepared.image_bgr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.images[index]
        if self.augment:
            image = _augment(image, self.rng)
        tensor = self.torch.from_numpy(cnn_input_tensor(image))
        return tensor, self.label_index[sample.label_id]


def _train_network(
    samples: list[GeometrySample],
    labels: list[str],
    epochs: int,
    seed: int,
    pretrained: bool,
    initial_state_dict=None,
    fine_tune_backbone: bool = False,
):
    torch, _ = _optional_imports(training=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _build_network(len(labels), pretrained)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    # With a very small labelled dataset, train the classifier head first. This
    # is both more stable than updating the whole ImageNet backbone and much
    # cheaper on the CPU-only development and deployment machines.
    if pretrained:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        if fine_tune_backbone:
            for block in model.features[-3:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
    dataset = _GeometryTorchDataset(samples, labels, augment=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(32, len(dataset)), shuffle=True, generator=generator
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4 if fine_tune_backbone else 8e-4,
        weight_decay=1e-4,
    )
    counts = Counter(sample.label_id for sample in samples)
    class_weights = torch.tensor(
        [len(samples) / (len(labels) * counts[label]) for label in labels],
        dtype=torch.float32,
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.05
    )
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    model.eval()
    return model, final_loss


def _predict_torch(
    model,
    samples: list[GeometrySample],
    labels: list[str],
    confidence_threshold: float = 0.65,
    margin_threshold: float = 0.12,
) -> list[str]:
    torch, _ = _optional_imports(training=True)
    predictions = ["unknown"] * len(samples)
    tensors: list[Any] = []
    valid_indices: list[int] = []
    for index, sample in enumerate(samples):
        prepared, _ = _prepare_object(sample.image_bgr, None, DEFAULT_INPUT_SIZE)
        if prepared is None:
            continue
        tensors.append(torch.from_numpy(cnn_input_tensor(prepared.image_bgr)))
        valid_indices.append(index)
    with torch.inference_mode():
        for start in range(0, len(tensors), 12):
            inputs = torch.stack(tensors[start : start + 12])
            probabilities = torch.softmax(model(inputs), dim=1)
            values, indices = probabilities.topk(min(2, len(labels)), dim=1)
            for row in range(len(inputs)):
                confidence = float(values[row, 0])
                second = float(values[row, 1]) if values.shape[1] > 1 else 0.0
                predictions[valid_indices[start + row]] = (
                    labels[int(indices[row, 0])]
                    if confidence >= confidence_threshold
                    and confidence - second >= margin_threshold
                    else "unknown"
                )
    return predictions


def _stratified_folds(samples: list[GeometrySample], folds: int, seed: int):
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.label_id, []).append(index)
    if any(len(indices) < folds for indices in grouped.values()):
        raise ValueError(f"each class requires at least {folds} samples for cross-validation")
    rng = random.Random(seed)
    fold_indices = [[] for _ in range(folds)]
    for indices in grouped.values():
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            fold_indices[offset % folds].append(index)
    return fold_indices


def load_cnn_training_samples(
    data_root: str | Path,
    additional_data_roots: list[str | Path] | None = None,
) -> tuple[list[GeometrySample], list[dict[str, str]], list[str], list[Path]]:
    """Load labelled batches while removing byte-identical camera captures."""
    roots = [Path(data_root), *(Path(item) for item in (additional_data_roots or []))]
    samples: list[GeometrySample] = []
    errors: list[dict[str, str]] = []
    duplicates: list[str] = []
    seen_hashes: set[str] = set()
    for root in roots:
        batch, batch_errors = load_geometry_samples(root)
        errors.extend(batch_errors)
        for sample in batch:
            if sample.sha256 in seen_hashes:
                duplicates.append(str(sample.path))
                continue
            seen_hashes.add(sample.sha256)
            samples.append(sample)
    return samples, errors, duplicates, roots


def train_geometry_cnn(
    data_root: str | Path,
    output: str | Path,
    epochs: int = 40,
    seed: int = 17,
    pretrained: bool = True,
    cross_validation: bool = True,
    additional_data_roots: list[str | Path] | None = None,
    resume_checkpoint: str | Path | None = None,
    fine_tune_backbone: bool = False,
) -> dict[str, Any]:
    torch, _ = _optional_imports(training=True)
    samples, errors, duplicates, roots = load_cnn_training_samples(
        data_root, additional_data_roots
    )
    if errors:
        raise ValueError(f"geometry dataset contains errors: {errors}")
    preprocessing_errors: list[dict[str, str]] = []
    usable_samples: list[GeometrySample] = []
    for sample in samples:
        prepared, reason = _prepare_object(
            sample.image_bgr, None, DEFAULT_INPUT_SIZE
        )
        if prepared is None:
            preprocessing_errors.append(
                {"path": str(sample.path), "reason": reason}
            )
            continue
        usable_samples.append(sample)
    samples = usable_samples
    labels = sorted(set(sample.label_id for sample in samples))
    class_names = {sample.label_id: sample.label_name for sample in samples}
    counts = Counter(sample.label_id for sample in samples)
    if len(labels) < 2 or any(count < 3 for count in counts.values()):
        raise ValueError("CNN training requires at least two classes and three samples per class")
    initial_state_dict = None
    previous_epochs = 0
    if resume_checkpoint is not None:
        try:
            previous = torch.load(
                Path(resume_checkpoint), map_location="cpu", weights_only=False
            )
        except TypeError:
            previous = torch.load(Path(resume_checkpoint), map_location="cpu")
        if previous.get("labels") != labels:
            raise ValueError("resume checkpoint labels do not match the training dataset")
        hashes = [sample.sha256 for sample in samples]
        if previous.get("source_hashes") != hashes:
            raise ValueError("resume checkpoint images do not match the training dataset")
        initial_state_dict = previous["state_dict"]
        previous_epochs = int(previous.get("trained_epochs", 0))
    fold_reports = []
    cross_validation_true: list[str] = []
    cross_validation_predicted: list[str] = []
    cross_validation_raw_predicted: list[str] = []
    if cross_validation:
        for fold, validation_indices in enumerate(_stratified_folds(samples, 3, seed)):
            validation = set(validation_indices)
            training_samples = [sample for index, sample in enumerate(samples) if index not in validation]
            validation_samples = [samples[index] for index in validation_indices]
            model, _ = _train_network(
                training_samples,
                labels,
                epochs,
                seed + fold,
                pretrained,
                fine_tune_backbone=fine_tune_backbone,
            )
            predicted = _predict_torch(model, validation_samples, labels)
            raw_predicted = _predict_torch(
                model,
                validation_samples,
                labels,
                confidence_threshold=0.0,
                margin_threshold=-1.0,
            )
            cross_validation_true.extend(sample.label_id for sample in validation_samples)
            cross_validation_predicted.extend(predicted)
            cross_validation_raw_predicted.extend(raw_predicted)
            accuracy = float(np.mean([
                prediction == sample.label_id
                for prediction, sample in zip(predicted, validation_samples)
            ]))
            raw_accuracy = float(np.mean([
                prediction == sample.label_id
                for prediction, sample in zip(raw_predicted, validation_samples)
            ]))
            fold_reports.append({
                "fold": fold + 1,
                "samples": len(validation_samples),
                "accepted_accuracy": accuracy,
                "raw_top1_accuracy": raw_accuracy,
            })
    matrix_labels = [*labels, "unknown"]
    matrix_index = {label: index for index, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(labels), len(matrix_labels)), np.int32)
    for truth, prediction in zip(cross_validation_true, cross_validation_predicted):
        matrix[matrix_index[truth], matrix_index.get(prediction, matrix_index["unknown"])] += 1
    recalls = {
        label: float(matrix[row, row] / max(matrix[row].sum(), 1))
        for row, label in enumerate(labels)
    }
    model, final_loss = _train_network(
        samples,
        labels,
        epochs,
        seed + previous_epochs,
        pretrained,
        initial_state_dict=initial_state_dict,
        fine_tune_backbone=fine_tune_backbone,
    )
    checkpoint = {
        "model_version": CNN_MODEL_VERSION,
        "architecture": "mobilenet_v3_small",
        "input_size": DEFAULT_INPUT_SIZE,
        "labels": labels,
        "class_names": class_names,
        "confidence_threshold": 0.65,
        "margin_threshold": 0.12,
        "state_dict": model.state_dict(),
        "source_hashes": [sample.sha256 for sample in samples],
        "trained_epochs": previous_epochs + epochs,
        "data_roots": [str(root) for root in roots],
        "same_batch_only": True,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return {
        "checkpoint": str(target.resolve()),
        "training_samples": len(samples),
        "data_roots": [str(root) for root in roots],
        "duplicate_samples_skipped": duplicates,
        "preprocessing_errors": preprocessing_errors,
        "class_counts": dict(sorted(counts.items())),
        "final_loss": final_loss,
        "trained_epochs": previous_epochs + epochs,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
        "training_strategy": (
            "imagenet_last_blocks_finetune"
            if pretrained and fine_tune_backbone
            else (
                "imagenet_frozen_backbone"
                if pretrained
                else "from_scratch_full_network"
            )
        ),
        "class_balancing": "inverse_frequency_cross_entropy",
        "cross_validation": fold_reports,
        "cross_validation_accuracy": float(np.mean([
            prediction == truth
            for prediction, truth in zip(
                cross_validation_predicted, cross_validation_true
            )
        ])) if fold_reports else None,
        "cross_validation_raw_top1_accuracy": float(np.mean([
            prediction == truth
            for prediction, truth in zip(
                cross_validation_raw_predicted, cross_validation_true
            )
        ])) if fold_reports else None,
        "cross_validation_labels": matrix_labels,
        "cross_validation_confusion_matrix": matrix.tolist(),
        "cross_validation_per_class_recall": recalls,
        "cross_validation_macro_recall": float(np.mean(list(recalls.values()))) if fold_reports else None,
        "cross_validation_rejection_rate": (
            cross_validation_predicted.count("unknown") / len(cross_validation_predicted)
            if cross_validation_predicted
            else None
        ),
        "same_batch_only": True,
    }


def _load_checkpoint(path: str | Path):
    torch, _ = _optional_imports(training=True)
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if checkpoint.get("model_version") != CNN_MODEL_VERSION:
        raise ValueError("unsupported geometry CNN checkpoint version")
    model = _build_network(len(checkpoint["labels"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _openvino_logits(compiled, tensors: np.ndarray) -> np.ndarray:
    result = compiled([tensors])
    if isinstance(result, Mapping):
        result = next(iter(result.values()))
    return np.asarray(result).reshape(len(tensors), -1)


def _dataset_tensors(data_root: str | Path, labels: list[str]):
    samples, errors = load_geometry_samples(data_root)
    if errors:
        raise ValueError(f"geometry dataset contains errors: {errors}")
    tensors = []
    targets = []
    label_index = {label: index for index, label in enumerate(labels)}
    for sample in samples:
        if sample.label_id not in label_index:
            continue
        prepared, reason = _prepare_object(sample.image_bgr, None, DEFAULT_INPUT_SIZE)
        if prepared is None:
            raise ValueError(f"cannot preprocess {sample.path}: {reason}")
        tensors.append(cnn_input_tensor(prepared.image_bgr))
        targets.append(label_index[sample.label_id])
    return np.stack(tensors), np.asarray(targets), samples


def export_geometry_cnn(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    precision: str = "int8",
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    if precision not in {"fp16", "fp32", "int8"}:
        raise ValueError("precision must be fp16, fp32 or int8")
    if precision == "int8" and data_root is None:
        raise ValueError("--data-root is required for INT8 calibration")
    torch, _ = _optional_imports(training=True)
    try:
        import openvino as ov
    except ImportError as error:
        raise RuntimeError("OpenVINO is required for CNN export") from error
    model, checkpoint = _load_checkpoint(checkpoint_path)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"CNN output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    onnx_path = target / "model.onnx"
    dummy = torch.zeros((1, 3, checkpoint["input_size"], checkpoint["input_size"]), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    ov_model = ov.convert_model(onnx_path)
    reference_path = target / ("model-fp32.xml" if precision == "fp32" else "model-fp16.xml")
    ov.save_model(ov_model, reference_path, compress_to_fp16=precision != "fp32")
    selected_file = reference_path.name
    quantization = {"requested": precision == "int8", "accepted": False, "accuracy_drop": None}
    if precision == "int8":
        try:
            import nncf
        except ImportError as error:
            raise RuntimeError("NNCF is required for INT8 calibration") from error
        tensors, targets, calibration_samples = _dataset_tensors(
            data_root, checkpoint["labels"]
        )
        calibration = nncf.Dataset([tensor[None, ...] for tensor in tensors])
        quantized = nncf.quantize(ov_model, calibration)
        int8_path = target / "model-int8.xml"
        ov.save_model(quantized, int8_path)
        core = ov.Core()
        reference_predictions = _openvino_logits(core.compile_model(reference_path, "CPU"), tensors).argmax(1)
        int8_predictions = _openvino_logits(core.compile_model(int8_path, "CPU"), tensors).argmax(1)
        reference_accuracy = float(np.mean(reference_predictions == targets))
        int8_accuracy = float(np.mean(int8_predictions == targets))
        drop = reference_accuracy - int8_accuracy
        quantization.update({
            "reference_accuracy": reference_accuracy,
            "int8_accuracy": int8_accuracy,
            "accuracy_drop": drop,
            "accepted": drop <= 0.02,
            "comparison": [
                {
                    "path": str(sample.path),
                    "true_label": checkpoint["labels"][int(target_index)],
                    "reference_prediction": checkpoint["labels"][int(reference_index)],
                    "int8_prediction": checkpoint["labels"][int(int8_index)],
                }
                for sample, target_index, reference_index, int8_index in zip(
                    calibration_samples,
                    targets,
                    reference_predictions,
                    int8_predictions,
                )
            ],
        })
        if drop <= 0.02:
            selected_file = int8_path.name
    metadata = {
        "model_version": CNN_MODEL_VERSION,
        "architecture": checkpoint["architecture"],
        "trained_epochs": int(checkpoint.get("trained_epochs", 0)),
        "data_roots": checkpoint.get("data_roots", []),
        "model_file": selected_file,
        "onnx_file": onnx_path.name,
        "precision_requested": precision,
        "precision_selected": "int8" if selected_file == "model-int8.xml" else (
            "fp32" if selected_file == "model-fp32.xml" else "fp16"
        ),
        "input_size": checkpoint["input_size"],
        "labels": checkpoint["labels"],
        "class_names": checkpoint["class_names"],
        "confidence_threshold": checkpoint["confidence_threshold"],
        "margin_threshold": checkpoint["margin_threshold"],
        "same_batch_only": True,
        "quantization": quantization,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"output_dir": str(target.resolve()), **metadata}


def evaluate_geometry_backend(
    data_root: str | Path, backend: str, model_path: str | Path, device: str = "CPU"
) -> dict[str, Any]:
    model = load_geometry_shape_model(backend, model_path, device)
    samples, errors = load_geometry_samples(data_root)
    rows = []
    labels = sorted(set(sample.label_id for sample in samples))
    matrix_labels = [*labels, "unknown"]
    index = {label: offset for offset, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(matrix_labels), len(matrix_labels)), np.int32)
    for sample in samples:
        prediction = model.predict_geometry(sample.image_bgr)
        matrix[index[sample.label_id], index.get(prediction.label_id, index["unknown"])] += 1
        rows.append({"path": str(sample.path), "true_label": sample.label_id, **prediction.to_dict()})
    accuracy = float(np.mean([row["true_label"] == row["label_id"] for row in rows])) if rows else 0.0
    recalls = {
        label: float(matrix[row, row] / max(matrix[row].sum(), 1))
        for row, label in enumerate(labels)
    }
    rejected = sum(row["label_id"] == "unknown" for row in rows)
    return {
        "backend": backend,
        "samples": len(rows),
        "accuracy": accuracy,
        "labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": recalls,
        "macro_recall": float(np.mean(list(recalls.values()))) if recalls else 0.0,
        "rejection_rate": rejected / max(len(rows), 1),
        "predictions": rows,
        "errors": errors,
        "same_batch_only": True,
        "warning": "当前数据来自同一采集批次，结果不能代表泛化能力。",
    }


def export_geometry_backend_results(
    data_root: str | Path,
    backend: str,
    model_path: str | Path,
    output_root: str | Path,
    device: str = "CPU",
) -> dict[str, Any]:
    target = Path(output_root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"geometry output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    model = load_geometry_shape_model(backend, model_path, device)
    samples, errors = load_geometry_samples(data_root)
    rows: list[dict[str, Any]] = []
    input_size = int(getattr(model, "input_size", 128))

    for sample in samples:
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=input_size)
        if prepared is None:
            errors.append({"path": str(sample.path), "reason": "object_not_found"})
            continue
        prediction = model.predict_geometry(sample.image_bgr)
        sample_dir = target / "按真实类别" / sample.label_name / sample.path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.path, sample_dir / "original.jpg")
        if not cv2.imwrite(str(sample_dir / "normalized.png"), prepared.image_bgr):
            raise OSError(f"failed to write normalized image: {sample.path}")
        if not cv2.imwrite(str(sample_dir / "mask.png"), prepared.mask):
            raise OSError(f"failed to write mask: {sample.path}")

        annotated = sample.image_bgr.copy()
        x, y, width, height = prepared.bbox_px
        colour = (0, 180, 0) if prediction.label_id == sample.label_id else (
            (0, 190, 255) if prediction.label_id == "unknown" else (0, 0, 220)
        )
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, 3)
        cv2.putText(
            annotated,
            f"{backend}: true={sample.label_id} pred={prediction.label_id} conf={prediction.confidence:.2f}",
            (max(5, x), max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(sample_dir / "annotated.jpg"), annotated):
            raise OSError(f"failed to write annotated image: {sample.path}")

        row = {
            "source_path": str(sample.path),
            "true_label": sample.label_id,
            "true_name": sample.label_name,
            **prediction.to_dict(),
            "correct": prediction.label_id == sample.label_id,
            "result_dir": str(sample_dir.relative_to(target)),
        }
        rows.append(row)
        (sample_dir / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    labels = sorted(set(sample.label_id for sample in samples))
    matrix_labels = [*labels, "unknown"]
    label_index = {label: index for index, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(labels), len(matrix_labels)), np.int32)
    for row in rows:
        matrix[
            label_index[row["true_label"]],
            label_index.get(row["label_id"], label_index["unknown"]),
        ] += 1
    recalls = {
        label: float(matrix[index, index] / max(matrix[index].sum(), 1))
        for index, label in enumerate(labels)
    }
    summary = {
        "backend": backend,
        "data_root": str(Path(data_root)),
        "model_path": str(Path(model_path)),
        "exported_images": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else 0.0,
        "rejection_rate": sum(row["label_id"] == "unknown" for row in rows) / max(len(rows), 1),
        "mean_inference_ms": float(np.mean([row["inference_ms"] for row in rows])) if rows else 0.0,
        "labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": recalls,
        "macro_recall": float(np.mean(list(recalls.values()))) if recalls else 0.0,
        "errors": errors,
        "same_batch_only": True,
        "warning": "当前图片参与了模型训练，结果不能代表对新批次和新角度的泛化能力。",
    }
    with (target / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (target / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *matrix_labels])
        for label, values in zip(labels, matrix):
            writer.writerow([label, *values.tolist()])
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "说明.txt").write_text(
        "\n".join(
            [
                f"几何分类结果（{backend}）",
                f"图片数量：{len(rows)}",
                f"同批次准确率：{summary['accuracy']:.2%}",
                f"拒识率：{summary['rejection_rate']:.2%}",
                "注意：当前图片参与了模型训练，本结果不能作为泛化性能或比赛验收。",
                "每张图片目录包含原图、标准化裁剪、掩膜、标注图和JSON结果。",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def benchmark_geometry_backend(
    data_root: str | Path,
    backend: str,
    model_path: str | Path,
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 200,
    device: str = "CPU",
) -> dict[str, Any]:
    if batch_size not in {1, 12}:
        raise ValueError("batch size must be 1 or 12")
    model = load_geometry_shape_model(backend, model_path, device)
    samples, errors = load_geometry_samples(data_root)
    if errors or not samples:
        raise ValueError(f"benchmark dataset is invalid: {errors}")
    batch = []
    input_size = int(getattr(model, "input_size", 128))
    for index in range(batch_size):
        sample = samples[index % len(samples)]
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=input_size)
        if prepared is None:
            raise ValueError(f"cannot preprocess benchmark sample: {sample.path}")
        batch.append((prepared.image_bgr, prepared.mask))

    def run_once():
        started = time.perf_counter()
        if hasattr(model, "predict_batch"):
            model.predict_batch(batch)
        else:
            for image, mask in batch:
                model.predict_geometry(image, mask)
        return (time.perf_counter() - started) * 1000.0

    for _ in range(warmup):
        run_once()
    durations = np.asarray([run_once() for _ in range(iterations)], np.float64)
    return {
        "backend": backend,
        "device": device,
        "batch_size": batch_size,
        "input": "presegmented_object_crops",
        "warmup": warmup,
        "iterations": iterations,
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "mean_ms": float(durations.mean()),
        "target_p95_ms": 30.0 if batch_size == 1 else 150.0,
        "meets_target": float(np.percentile(durations, 95)) <= (30.0 if batch_size == 1 else 150.0),
    }
