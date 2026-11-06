import csv
from os import path
from sys import argv
from json import load
from dataclasses import dataclass
from app.components.ArgumentsParser import ArgumentsParser
from hooks.check_dict_compatibility import check_dict_compatibility
from hooks.wilcoxon_tests import wilcoxon_tests


@dataclass
class AnalysisSource:
    label: str
    values: list[float]


@dataclass
class AnalysisData:
    target: list[float]
    sources: list[AnalysisSource]

    def map_sources(self):
        return [ source.values for source in self.sources ]

    def map_labels(self):
        return [ source.label for source in self.sources ]


def map_source(segment):
    if check_dict_compatibility(segment, AnalysisData):
        return AnalysisData(**segment)
    if check_dict_compatibility(segment, AnalysisSource):
        return AnalysisSource(**segment)

    return None


if __name__ == "__main__":
    arguments_parser = ArgumentsParser(argv[1:])
    arguments_parser.parse()

    source_path = arguments_parser.get_option("source", str)

    try:
        assert isinstance(source_path, str)
        assert path.exists(source_path) and path.isfile(source_path)
    except AssertionError:
        print("Incorrect arguments specified. You ned to specify: --source=<existing-file-path>")
        exit(1)

    with open(source_path) as file:
        data = load(file, object_hook=map_source)
        assert isinstance(data, AnalysisData)

    results = wilcoxon_tests(data.target, *data.map_sources(), alpha=0.05, method="holm")

    for result, label in zip(results, data.map_labels()):
        print(f"For {label} ({'rejected' if result.rejected else 'approved'}): ")
        print("")
        print(f"Delta: {-result.delta * 100:.2f}")
        print(f"Statistic: {result.T:.1f}")
        print(f"p (Holm): {result.p_holm:.3f}")
        print("")
        print("")

    csv_path = input("Give an output CSV file name: ")

    with open(csv_path, "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")

        csv_writer.writerow([ "Variant", "delta MAP", "T", "p (Holm)" ])

        for result, label in zip(results, data.map_labels()):
            csv_writer.writerow([
                label,
                f"{-result.delta * 100:.2f}".replace(".", ","),
                f"{result.T:.1f}".replace(".", ","),
                f"{result.p_holm:.3f}".replace(".", ",")
            ])
