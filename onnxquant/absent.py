from pydantic import BaseModel, ConfigDict


class NotComputed(BaseModel):
    """Marks a report section that was legitimately skipped, with why.

    Used instead of a bare None wherever a report can be built without running an
    optional, more expensive step (e.g. no validation samples on hand for parity,
    or a debugging pass the caller chose not to run).
    """

    model_config = ConfigDict(frozen=True)

    reason: str
