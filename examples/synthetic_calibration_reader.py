import numpy as np

from onnxquant import CalibrationDataReader


class SyntheticCalibrationReader(CalibrationDataReader):
    """Stand-in for a real calibration data source.

    onnxquant never loads data itself — implement get_next() against your own
    dataset (files, a dataloader, cloud storage, whatever) and it plugs in the
    same way this synthetic one does.
    """

    def __init__(
        self,
        input_name: str,
        input_shape: tuple[int, ...],
        num_samples: int,
        seed: int = 0,
    ) -> None:
        self._input_name = input_name
        self._input_shape = input_shape
        self._rng = np.random.default_rng(seed)
        self._remaining = num_samples

    # CalibrationDataReader.get_next is stubbed as `-> dict`, but its own __next__
    # explicitly treats a None return as "exhausted" (onnxruntime's real contract,
    # just not reflected in the stub) — this override matches the real contract.
    def get_next(self) -> dict[str, np.ndarray] | None:  # pyrefly: ignore[bad-override]
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        sample = self._rng.standard_normal(self._input_shape).astype(np.float32)
        return {self._input_name: sample}
