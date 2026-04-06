from typing import Any

from lib.computation.classifiers import (
    Classifier,
    LDAClassifier,
    SklearnClassifier,
)



def load_classifier(classifier: str, **kwargs: Any) -> Classifier:
    if classifier.split("_")[0] == "sklearn":
        if "device" in kwargs.keys():
            kwargs.pop("device")
        return SklearnClassifier(type=classifier, **kwargs)
    match classifier:
        case "lda":
            return LDAClassifier(**kwargs)
        case _:
            raise ValueError(f"Invalid classifier: {classifier}")


def list_all_classifiers() -> list[str]:
    return ["lda", "sklearn_lda", "sklearn_qda", "sklearn_logistic", "sklearn_svm", "sklearn_tree", "sklearn_nb", "sklearn_gp"]
