from typing import TypedDict, Literal
from dataclasses import dataclass
from numpy import array, round
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


@dataclass
class WilcoxonResult:
    rejected: float
    delta: float
    T: float
    p: float
    p_holm: float

    def set_rejected(self, rejected):
        self.rejected = rejected
        return self

    def set_p_holm(self, p_holm):
        self.p_holm = p_holm
        return self


class WilcoxonOptions(TypedDict):
    alpha: float
    method: Literal["auto", "two-sided", "wilcox"]


def wilcoxon_test(target, source):
    if len(target) != len(source):
        raise ValueError("The length of target and source must be same.")

    difference = round(array(target) - array(source), 8)

    result = wilcoxon(
        difference,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    return WilcoxonResult(
        rejected=False,
        delta=difference.mean(),
        T=result.statistic,
        p=result.pvalue,
        p_holm=result.pvalue
    )


def wilcoxon_tests(target, *sources, **options):
    results = [ wilcoxon_test(target, source) for source in sources ]

    rejected_results, p_holm_results, _, _ = multipletests(
        [ result.p for result in results ],
        alpha=options["alpha"],
        method=options["method"]
    )

    return [
        result.set_rejected(rejected).set_p_holm(p_holm)
        for result, rejected, p_holm in zip(results, rejected_results, p_holm_results)
    ]
