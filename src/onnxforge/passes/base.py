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

    @abstractmethod
    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        """Apply the pass. Must not mutate the input model. Returns new model + result."""
        ...

    def is_applicable(self, model: onnx.ModelProto) -> bool:
        return True
