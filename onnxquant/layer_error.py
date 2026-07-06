import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors,
)
from onnxruntime.quantization.qdq_loss_debug import (
    compute_signal_to_quantization_noice_ratio as compute_sqnr,
)
from pydantic import BaseModel

from onnxquant.absent import NotComputed
from onnxquant.preprocess import preprocess
from onnxquant.protocols import CalibrationDataReader


class WeightError(BaseModel):
    weight_name: str
    sqnr_db: float


class ActivationError(BaseModel):
    tensor_name: str
    qdq_err_db: float
    xmodel_err_db: float | NotComputed


class LayerErrorReport(BaseModel):
    """Per-tensor SQNR (dB, higher is better) — where a decomposed graph loses precision.

    Distinct from ParityReport: that compares final outputs only, this locates the
    loss to a specific weight or activation tensor.
    """

    weights: list[WeightError]
    activations: list[ActivationError]

    def to_markdown(self) -> str:
        lines = ["# Layer Error Report (SQNR, dB — higher is better)", "", "## Weights"]
        lines += ["| weight | sqnr_db |", "|---|---|"]
        for weight_error in sorted(self.weights, key=lambda w: w.sqnr_db):
            lines.append(f"| {weight_error.weight_name} | {weight_error.sqnr_db:.2f} |")

        lines += ["", "## Activations"]
        lines += ["| tensor | qdq_err_db | xmodel_err_db |", "|---|---|---|"]
        for activation_error in sorted(self.activations, key=lambda a: a.qdq_err_db):
            xmodel_err = activation_error.xmodel_err_db
            xmodel_str = (
                f"not computed: {xmodel_err.reason}"
                if isinstance(xmodel_err, NotComputed)
                else f"{xmodel_err:.2f}"
            )
            lines.append(
                f"| {activation_error.tensor_name} | {activation_error.qdq_err_db:.2f} "
                f"| {xmodel_str} |"
            )

        return "\n".join(lines)


def compute_layer_error(
    fp32_model: str | Path,
    int8_model: str | Path,
    make_samples: Callable[[], CalibrationDataReader],
) -> LayerErrorReport:
    """Locate quantization-induced precision loss to specific weight/activation tensors.

    `make_samples` is a zero-arg factory rather than a single reader/iterable because
    this needs two independent passes over the same distribution — one run against the
    float model, one against the quantized model — so a single-use reader can't serve
    both.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        # Preprocessing changes tensor names (fusion, constant folding); int8_model was
        # quantized from a preprocessed graph, so comparing against the *original*
        # fp32_model would misalign tensor names between the two augmented models below.
        preprocessed_fp32_path = tmp / "preprocessed_fp32.onnx"
        preprocess(fp32_model, preprocessed_fp32_path)

        weight_matching = create_weight_matching(str(preprocessed_fp32_path), str(int8_model))
        weight_errors = [
            WeightError(weight_name=name, sqnr_db=sqnr)
            for name, sqnr in compute_weight_error(weight_matching).items()
        ]

        augmented_fp32_path = tmp / "augmented_fp32.onnx"
        augmented_int8_path = tmp / "augmented_int8.onnx"
        modify_model_output_intermediate_tensors(preprocessed_fp32_path, augmented_fp32_path)
        modify_model_output_intermediate_tensors(int8_model, augmented_int8_path)

        float_activations = collect_activations(str(augmented_fp32_path), make_samples())
        qdq_activations = collect_activations(str(augmented_int8_path), make_samples())

    # collect_activations() is typed to return dict[str, list[ndarray]], but dict is
    # invariant in its value type, so it doesn't structurally satisfy create_activation_
    # matching()'s dict[str, Sequence[ndarray]] param even though list is a Sequence.
    # Stub imprecision, not a real type hazard.
    activation_matching = create_activation_matching(
        cast("dict[str, Sequence[np.ndarray]]", qdq_activations),
        cast("dict[str, Sequence[np.ndarray]]", float_activations),
    )
    # Not using onnxruntime's own compute_activation_error() here: it indexes
    # match["float"] unconditionally, which raises KeyError for any tensor create_
    # activation_matching() couldn't pair with a float activation — a real possibility,
    # not just a hypothetical, so we do the lookup defensively ourselves instead.
    activation_errors = [
        ActivationError(
            tensor_name=name,
            qdq_err_db=compute_sqnr(match["pre_qdq"], match["post_qdq"]),
            xmodel_err_db=(
                compute_sqnr(match["float"], match["post_qdq"])
                if "float" in match
                else NotComputed(reason="no matching float activation tensor found")
            ),
        )
        for name, match in activation_matching.items()
    ]

    return LayerErrorReport(weights=weight_errors, activations=activation_errors)
