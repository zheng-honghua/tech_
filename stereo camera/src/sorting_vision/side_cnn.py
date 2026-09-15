from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .geometry_cnn import (
    DEFAULT_INPUT_SIZE,
    _build_network,
    _optional_imports,
    cnn_input_tensor,
)
from .geometry_rgb import preprocess_geometry_object
from .shape_registry import ShapeRegistry
from .side_geometry import SideTrainingSample
from .synthetic_side import render_synthetic_side


SIDE_CNN_VERSION = 1


def _prepare(image: np.ndarray, mask: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    prepared = preprocess_geometry_object(image, supplied_mask=mask, output_size=size)
    if prepared is None:
        raise ValueError("cannot preprocess side-view object for CNN")
    return cnn_input_tensor(prepared.image_bgr)


class _Dataset:
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, str]], labels: tuple[str, ...], seed: int):
        torch, _ = _optional_imports(training=True)
        self.torch = torch
        self.samples = samples
        self.label_index = {label: index for index, label in enumerate(labels)}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, mask, label = self.samples[index]
        tensor = _prepare(image, mask)
        if self.rng.random() < 0.5:
            tensor = tensor[:, :, ::-1].copy()
        return self.torch.from_numpy(tensor), self.label_index[label]


def _train_phase(model: Any, samples: list[tuple[np.ndarray, np.ndarray, str]], labels: tuple[str, ...], epochs: int, seed: int) -> float:
    torch, _ = _optional_imports(training=True)
    if not samples or epochs <= 0:
        return 0.0
    dataset = _Dataset(samples, labels, seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(32, len(dataset)), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=8e-4, weight_decay=1e-4,
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    return final_loss


def train_side_cnn(
    samples: Iterable[SideTrainingSample],
    registry: ShapeRegistry,
    output: str | Path,
    *,
    epochs: int = 30,
    seed: int = 23,
    synthetic_pretrain_per_class: int = 0,
) -> dict[str, Any]:
    """Train a frozen ImageNet MobileNetV3-Small side branch on CPU-compatible code."""
    torch, _ = _optional_imports(training=True)
    real = list(samples)
    labels = registry.class_ids
    if set(item.label_id for item in real) != set(labels):
        raise ValueError("side CNN training requires every enabled registry class")
    model = _build_network(len(labels), pretrained=True)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    synthetic: list[tuple[np.ndarray, np.ndarray, str]] = []
    for class_index, label in enumerate(labels):
        for repeat in range(max(0, synthetic_pretrain_per_class)):
            image, mask = render_synthetic_side(label, seed=seed + class_index * 1000 + repeat)
            synthetic.append((image, mask, label))
    synthetic_loss = _train_phase(model, synthetic, labels, max(1, epochs // 4), seed) if synthetic else None
    real_values = [(item.image_bgr, item.mask, item.label_id) for item in real]
    real_loss = _train_phase(model, real_values, labels, epochs, seed + 1)
    model.eval()
    checkpoint = {
        "model_type": "side_mobilenet_v3_small",
        "model_version": SIDE_CNN_VERSION,
        "architecture": "mobilenet_v3_small",
        "input_size": DEFAULT_INPUT_SIZE,
        "class_ids": list(labels),
        "class_names": registry.names,
        "registry_hash": registry.registry_hash,
        "confidence_threshold": 0.65,
        "margin_threshold": 0.12,
        "state_dict": model.state_dict(),
        "source_hashes": [item.sha256 for item in real],
        "synthetic_pretrain_samples": len(synthetic),
        "synthetic_used_for_calibration": False,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return {
        "checkpoint": str(target.resolve()),
        "registry_hash": registry.registry_hash,
        "real_training_samples": len(real),
        "synthetic_pretrain_samples": len(synthetic),
        "synthetic_loss": synthetic_loss,
        "real_loss": real_loss,
        "architecture": "MobileNetV3-Small",
        "input_size": DEFAULT_INPUT_SIZE,
        "backbone": "ImageNet pretrained, frozen",
        "formal_calibration_data": "real_only",
    }


def export_side_cnn(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    target_backend: str = "both",
) -> dict[str, Any]:
    if target_backend not in {"openvino", "tensorrt", "both"}:
        raise ValueError("target_backend must be openvino, tensorrt or both")
    torch, _ = _optional_imports(training=True)
    try:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if checkpoint.get("model_type") != "side_mobilenet_v3_small":
        raise ValueError("not a side MobileNetV3 checkpoint")
    model = _build_network(len(checkpoint["class_ids"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"side CNN output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    onnx_path = target / "side-model.onnx"
    dummy = torch.zeros((1, 3, checkpoint["input_size"], checkpoint["input_size"]), dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_path, input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    openvino_file = None
    if target_backend in {"openvino", "both"}:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError("OpenVINO export requested but openvino is not installed") from error
        model_ir = ov.convert_model(onnx_path)
        openvino_path = target / "side-model-fp16.xml"
        ov.save_model(model_ir, openvino_path, compress_to_fp16=True)
        openvino_file = openvino_path.name
    engine_file = "side-model-fp16.engine" if target_backend in {"tensorrt", "both"} else None
    trtexec_command = None
    if engine_file:
        size = int(checkpoint["input_size"])
        trtexec_command = (
            f"trtexec --onnx={onnx_path.name} --saveEngine={engine_file} --fp16 "
            f"--minShapes=images:1x3x{size}x{size} --optShapes=images:1x3x{size}x{size} "
            f"--maxShapes=images:8x3x{size}x{size}"
        )
        (target / "BUILD_TENSORRT_ON_ORIN.txt").write_text(
            "Run on the target Orin Nano from this directory:\n" + trtexec_command + "\n",
            encoding="utf-8",
        )
    metadata = {
        "schema_version": 1,
        "model_type": "side_mobilenet_v3_small",
        "model_version": SIDE_CNN_VERSION,
        "architecture": checkpoint["architecture"],
        "input_size": checkpoint["input_size"],
        "class_ids": checkpoint["class_ids"],
        "class_names": checkpoint["class_names"],
        "registry_hash": checkpoint["registry_hash"],
        "confidence_threshold": checkpoint["confidence_threshold"],
        "margin_threshold": checkpoint["margin_threshold"],
        "onnx_file": onnx_path.name,
        "openvino_file": openvino_file,
        "tensorrt_engine_file": engine_file,
        "tensorrt_build_command": trtexec_command,
        "precision": "fp16",
    }
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(target.resolve()), **metadata}


def build_tensorrt_engine(model_dir: str | Path, trtexec: str = "trtexec") -> Path:
    directory = Path(model_dir).resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    engine = directory / str(metadata["tensorrt_engine_file"])
    command = [
        trtexec,
        f"--onnx={directory / metadata['onnx_file']}",
        f"--saveEngine={engine}",
        "--fp16",
        f"--minShapes=images:1x3x{metadata['input_size']}x{metadata['input_size']}",
        f"--optShapes=images:1x3x{metadata['input_size']}x{metadata['input_size']}",
        f"--maxShapes=images:8x3x{metadata['input_size']}x{metadata['input_size']}",
    ]
    subprocess.run(command, check=True, cwd=directory)
    if not engine.is_file():
        raise RuntimeError("trtexec completed without writing the requested engine")
    return engine


class SideCNNModel:
    def __init__(self, infer: Any, metadata: dict[str, Any], registry: ShapeRegistry, backend: str):
        registry.validate_model_classes(metadata["class_ids"], metadata["registry_hash"])
        self.infer = infer
        self.metadata = metadata
        self.registry = registry
        self.registry_hash = registry.registry_hash
        self.backend = backend
        self.class_ids = tuple(metadata["class_ids"])
        self.last_class_scores: dict[str, float] = {}
        self.last_feature_diagnostics: dict[str, float] = {}

    @classmethod
    def load(cls, path: str | Path, backend: str, registry: ShapeRegistry, device: str = "CPU") -> "SideCNNModel":
        directory = Path(path)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if backend == "opencv":
            net = cv2.dnn.readNetFromONNX(str(directory / metadata["onnx_file"]))

            def infer(batch: np.ndarray) -> np.ndarray:
                net.setInput(batch)
                return np.asarray(net.forward())

            return cls(infer, metadata, registry, backend)
        if backend == "openvino":
            try:
                import openvino as ov
            except ImportError as error:
                raise RuntimeError("OpenVINO is not installed") from error
            model_file = metadata.get("openvino_file")
            if not model_file:
                raise FileNotFoundError("export does not contain an OpenVINO model")
            compiled = ov.Core().compile_model(directory / model_file, device)

            def infer(batch: np.ndarray) -> np.ndarray:
                output = compiled([batch])
                return np.asarray(next(iter(output.values())) if isinstance(output, Mapping) else output)

            return cls(infer, metadata, registry, backend)
        if backend == "tensorrt":
            engine = directory / str(metadata.get("tensorrt_engine_file", ""))
            if not engine.is_file():
                raise FileNotFoundError(
                    "TensorRT engine is missing; build it on the Orin Nano with dual-side-tensorrt-build"
                )
            try:
                from .tensorrt_runtime import TensorRTCallable
            except (ImportError, OSError) as error:
                raise RuntimeError("TensorRT/pycuda runtime is unavailable on this host") from error
            return cls(TensorRTCallable(engine), metadata, registry, backend)
        raise ValueError(f"unsupported side CNN backend: {backend}")

    def predict(self, image_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float, dict[str, Any]]:
        if mask is None:
            return "unknown", 0.0, {"reason": "side_mask_required"}
        started = time.perf_counter()
        try:
            tensor = _prepare(image_bgr, mask, int(self.metadata["input_size"]))
        except ValueError as error:
            self.last_class_scores = {}
            return "unknown", 0.0, {"reason": "object_not_found", "detail": str(error)}
        logits = np.asarray(self.infer(tensor[None, ...]), np.float32).reshape(-1)
        if len(logits) != len(self.class_ids):
            raise ValueError("side CNN output class count differs from registry")
        probabilities = np.exp(logits - float(logits.max()))
        probabilities /= max(float(probabilities.sum()), 1e-12)
        self.last_class_scores = {label: float(probabilities[index]) for index, label in enumerate(self.class_ids)}
        ordered = sorted(self.last_class_scores.items(), key=lambda item: item[1], reverse=True)
        label, probability = ordered[0]
        margin = probability - ordered[1][1]
        threshold = float(self.metadata["confidence_threshold"])
        margin_threshold = float(self.metadata["margin_threshold"])
        reason = "accepted" if probability >= threshold and margin >= margin_threshold else (
            "confidence_rejected" if probability < threshold else "margin_rejected"
        )
        self.last_feature_diagnostics = {
            "cnn_input_size": float(self.metadata["input_size"]),
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }
        return (label if reason == "accepted" else "unknown"), probability, {
            "nearest_label": label, "reason": reason, "margin": margin,
            "feature_groups": self.last_feature_diagnostics,
        }


def compare_side_cnn_backends(
    model_dir: str | Path,
    samples: Iterable[SideTrainingSample],
    registry: ShapeRegistry,
    backends: Iterable[str] = ("opencv", "openvino", "tensorrt"),
    device: str = "CPU",
) -> dict[str, Any]:
    """Check class/acceptance parity and P95 latency on the same reviewed images."""
    items = list(samples)
    requested = tuple(backends)
    if not items:
        raise ValueError("no reviewed side samples for backend comparison")
    outputs: dict[str, list[dict[str, Any]]] = {}
    unavailable: dict[str, str] = {}
    latency: dict[str, float] = {}
    for backend in requested:
        try:
            model = SideCNNModel.load(model_dir, backend, registry, device=device)
        except (ImportError, OSError, RuntimeError, FileNotFoundError, cv2.error) as error:
            unavailable[backend] = str(error)
            continue
        rows: list[dict[str, Any]] = []
        values: list[float] = []
        for item in items:
            started = time.perf_counter()
            label, confidence, diagnostics = model.predict(item.image_bgr, item.mask)
            values.append((time.perf_counter() - started) * 1000.0)
            rows.append(
                {
                    "sample": str(item.directory),
                    "label": label,
                    "accepted": diagnostics.get("reason") == "accepted",
                    "confidence": confidence,
                }
            )
        outputs[backend] = rows
        latency[backend] = float(np.percentile(values, 95))
    reference_backend = "opencv" if "opencv" in outputs else next(iter(outputs), None)
    mismatches: list[dict[str, Any]] = []
    if reference_backend is not None:
        reference = outputs[reference_backend]
        for backend, rows in outputs.items():
            if backend == reference_backend:
                continue
            for expected, actual in zip(reference, rows):
                if (expected["label"], expected["accepted"]) != (actual["label"], actual["accepted"]):
                    mismatches.append(
                        {
                            "sample": expected["sample"],
                            "reference_backend": reference_backend,
                            "reference_label": expected["label"],
                            "reference_accepted": expected["accepted"],
                            "backend": backend,
                            "label": actual["label"],
                            "accepted": actual["accepted"],
                        }
                    )
    return {
        "schema_version": 1,
        "registry_hash": registry.registry_hash,
        "sample_count": len(items),
        "requested_backends": list(requested),
        "tested_backends": list(outputs),
        "unavailable_backends": unavailable,
        "latency_p95_ms": latency,
        "mismatches": mismatches,
        "classes_and_acceptance_consistent": bool(outputs) and not mismatches,
        "all_requested_backends_tested": all(item in outputs for item in requested),
        "latency_gate_passed": all(value <= 1000.0 for value in latency.values()) and bool(latency),
        "passed": (
            all(item in outputs for item in requested)
            and not mismatches
            and all(value <= 1000.0 for value in latency.values())
        ),
    }


def calibrate_side_cnn_acceptance(
    model_dir: str | Path,
    samples: Iterable[SideTrainingSample],
    registry: ShapeRegistry,
    *,
    backend: str = "opencv",
    device: str = "CPU",
) -> dict[str, Any]:
    """Fit shared CNN confidence/margin gates on reviewed real calibration images."""
    items = list(samples)
    if not items:
        raise ValueError("no real side CNN calibration samples")
    if set(item.label_id for item in items) != set(registry.class_ids):
        raise ValueError("side CNN calibration requires every enabled registry class")
    directory = Path(model_dir)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry.validate_model_classes(metadata["class_ids"], metadata["registry_hash"])
    model = SideCNNModel.load(directory, backend, registry, device=device)
    observations: list[tuple[str, str, float, float]] = []
    for item in items:
        model.predict(item.image_bgr, item.mask)
        ordered = sorted(model.last_class_scores.items(), key=lambda value: value[1], reverse=True)
        if len(ordered) < 2:
            raise ValueError(f"CNN emitted fewer than two classes for {item.directory}")
        observations.append(
            (item.label_id, ordered[0][0], float(ordered[0][1]), float(ordered[0][1] - ordered[1][1]))
        )
    confidence_candidates = sorted(
        set([0.0, 1.0, *[item[2] for item in observations], *map(float, np.linspace(0.45, 0.90, 19))])
    )
    margin_candidates = sorted(
        set([0.0, *[item[3] for item in observations], *map(float, np.linspace(0.02, 0.40, 20))])
    )
    labels = registry.class_ids
    candidates: list[dict[str, Any]] = []
    for confidence_threshold in confidence_candidates:
        for margin_threshold in margin_candidates:
            predictions = [
                predicted if confidence >= confidence_threshold and margin >= margin_threshold else None
                for _, predicted, confidence, margin in observations
            ]
            wrong = sum(
                prediction is not None and prediction != truth
                for prediction, (truth, _, _, _) in zip(predictions, observations)
            )
            recalls = []
            for label in labels:
                indices = [index for index, value in enumerate(observations) if value[0] == label]
                recalls.append(sum(predictions[index] == label for index in indices) / max(len(indices), 1))
            candidates.append(
                {
                    "confidence_threshold": float(confidence_threshold),
                    "margin_threshold": float(margin_threshold),
                    "wrong_accept_count": int(wrong),
                    "macro_recall": float(np.mean(recalls)),
                    "coverage": sum(item is not None for item in predictions) / len(predictions),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["wrong_accept_count"] == 0,
            -item["wrong_accept_count"],
            item["macro_recall"],
            item["coverage"],
            item["confidence_threshold"],
            item["margin_threshold"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    metadata.update(
        {
            "confidence_threshold": selected["confidence_threshold"],
            "margin_threshold": selected["margin_threshold"],
            "acceptance_calibration": {
                "split": "probability_calibration",
                "sample_count": len(items),
                "real_images_only": True,
                "source_hashes": [item.sha256 for item in items],
                "selection_priority": ["zero_wrong_accept", "macro_recall", "coverage"],
                "selected": selected,
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"schema_version": 1, "model_dir": str(directory.resolve()), **metadata["acceptance_calibration"]}
