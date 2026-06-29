from abc import ABC, abstractmethod
from dataclasses import dataclass

import onnx


@dataclass
class PassResult:
    applied: bool
    description: str
    nodes_before: int
    nodes_after: int
    error: str | None = None


class Pass(ABC):
    name: str
    description: str
    # Minimum standard-domain ONNX opset the pass requires.  The pipeline
    # checks this before calling is_applicable so passes never need to
    # duplicate the check themselves.  Default 1 = any opset.
    min_std_opset: int = 1

    @abstractmethod
    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        """Apply the pass. Must not mutate the input model. Returns new model + result."""
        ...

    def is_applicable(self, model: onnx.ModelProto) -> bool:
        return True
