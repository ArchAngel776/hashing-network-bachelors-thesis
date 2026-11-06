from typing import Self, TypedDict, Literal
from dataclasses import dataclass


@dataclass
class WilcoxonResult:
    rejected: bool
    delta: float
    T: float
    p: float
    p_holm: float

    def set_rejected(self: Self, rejected: bool) -> Self: ...

    def set_p_holm(self: Self, p_holm: float) -> Self: ...


class WilcoxonOptions(TypedDict):
    alpha: float
    method: Literal["auto", "two-sided", "wilcox", "holm"]


def wilcoxon_test(target: list[float], source: list[float]) -> WilcoxonResult: ...

def wilcoxon_tests(target: list[float], *sources: list[float], **options) -> list[
    WilcoxonResult
]: ...
