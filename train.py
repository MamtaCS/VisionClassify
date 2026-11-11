from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from vision_utils import IMAGE_SIZE, extract_features

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DataSet"
RESULT_DIR = BASE_DIR / "result"
MODEL_PATH = RESULT_DIR / "best_model.joblib"
NEAR_DUPLICATE_DISTANCE = 3
MIN_ACCEPTED_ACCURACY = 0.99
MIN_ACCEPTED_COVERAGE = 0.50


def perceptual_hash(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (32, 32)).astype(np.float32)
    low_frequencies = cv2.dct(resized)[:8, :8]
    threshold = np.median(low_frequencies.flatten()[1:])
    return np.packbits((low_frequencies > threshold).flatten())


def build_groups(
    images: list[np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    hashes = np.asarray([perceptual_hash(image) for image in images], dtype=np.uint8)
    differences = np.unpackbits(
        np.bitwise_xor(hashes[:, None, :], hashes[None, :, :]),
        axis=2,
    ).sum(axis=2)

    parents = list(range(len(labels)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    cross_class_pairs = 0
    for left in range(len(labels)):
        for right in np.flatnonzero(
            differences[left, left + 1 :] <= NEAR_DUPLICATE_DISTANCE
        ) + left + 1:
            if labels[left] != labels[right]:
                cross_class_pairs += 1
                continue
            left_root = find(left)
            right_root = find(int(right))
            if left_root != right_root:
                parents[right_root] = left_root

    group_ids: dict[int, int] = {}
    groups = []
    for index in range(len(labels)):
        root = find(index)
        if root not in group_ids:
            group_ids[root] = len(group_ids)
        groups.append(group_ids[root])
    return np.asarray(groups), cross_class_pairs


def choose_score_threshold(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> float:
    eligible_thresholds = []
    for threshold in np.arange(0.20, 0.81, 0.05):
        accepted = scores >= threshold
        if not np.any(accepted) or np.mean(accepted) < MIN_ACCEPTED_COVERAGE:
            continue
        accepted_accuracy = accuracy_score(labels[accepted], predictions[accepted])
        if accepted_accuracy >= MIN_ACCEPTED_ACCURACY:
            eligible_thresholds.append(float(threshold))
    return min(eligible_thresholds) if eligible_thresholds else 0.50


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    images: list[np.ndarray] = []
    labels_list: list[str] = []
    class_counts: dict[str, int] = {}

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        loaded = 0
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            images.append(image)
            labels_list.append(class_dir.name)
            loaded += 1
        class_counts[class_dir.name] = loaded

    labels = np.asarray(labels_list)
    if len(set(labels)) != 10:
        raise ValueError(
            f"Expected the existing 10 object categories, found {len(set(labels))}: "
            f"{sorted(set(labels))}"
        )
    if any(count == 0 for count in class_counts.values()):
        raise ValueError(f"Every class must contain readable images: {class_counts}")

    features = np.asarray(
        [extract_features(image) for image in images],
        dtype=np.float32,
    )
    groups, cross_class_pairs = build_groups(images, labels)
    group_counts = {
        label: len(set(groups[labels == label]))
        for label in sorted(set(labels))
    }
    if min(group_counts.values()) < 5:
        raise ValueError(
            "Each class needs at least five distinct near-duplicate groups "
            f"for grouped cross-validation: {group_counts}"
        )

    print("Images per class:", class_counts)
    print(f"Total readable images: {len(labels)}")
    print(f"Near-duplicate groups: {len(set(groups))}")
    print(f"Cross-class near-duplicate pairs (review labels): {cross_class_pairs}")

    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    development_indices, test_indices = next(
        outer_cv.split(features, labels, groups)
    )
    development_features = features[development_indices]
    development_labels = labels[development_indices]
    development_groups = groups[development_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    candidates = {}
    for tree_count in (100, 300):
        out_of_fold_predictions = np.empty(len(development_labels), dtype=object)
        out_of_fold_scores = np.zeros(len(development_labels), dtype=np.float64)
        fold_accuracies = []
        fold_f1_scores = []

        for fit_indices, validation_indices in inner_cv.split(
            development_features,
            development_labels,
            development_groups,
        ):
            model = RandomForestClassifier(
                n_estimators=tree_count,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                development_features[fit_indices],
                development_labels[fit_indices],
            )
            predictions = model.predict(development_features[validation_indices])
            probabilities = model.predict_proba(
                development_features[validation_indices]
            )
            out_of_fold_predictions[validation_indices] = predictions
            out_of_fold_scores[validation_indices] = probabilities.max(axis=1)
            fold_accuracies.append(
                accuracy_score(
                    development_labels[validation_indices],
                    predictions,
                )
            )
            fold_f1_scores.append(
                f1_score(
                    development_labels[validation_indices],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

        candidates[tree_count] = {
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "mean_macro_f1": float(np.mean(fold_f1_scores)),
            "predictions": out_of_fold_predictions,
            "scores": out_of_fold_scores,
        }
        print(
            f"{tree_count}-tree grouped CV: "
            f"accuracy={candidates[tree_count]['mean_accuracy']:.2%}, "
            f"macro-F1={candidates[tree_count]['mean_macro_f1']:.2%}"
        )

    selected_tree_count = max(
        candidates,
        key=lambda count: candidates[count]["mean_macro_f1"],
    )
    selected = candidates[selected_tree_count]
    score_threshold = choose_score_threshold(
        development_labels,
        selected["predictions"],
        selected["scores"],
    )
    validation_accepted = selected["scores"] >= score_threshold
    validation_coverage = float(np.mean(validation_accepted))
    validation_accuracy = accuracy_score(
        development_labels[validation_accepted],
        selected["predictions"][validation_accepted],
    )
    print(f"Selected model: Random Forest ({selected_tree_count} trees)")
    print(
        f"Validation score threshold: {score_threshold:.2f}; "
        f"coverage={validation_coverage:.2%}, "
        f"accepted accuracy={validation_accuracy:.2%}"
    )

    test_model = RandomForestClassifier(
        n_estimators=selected_tree_count,
        random_state=42,
        n_jobs=-1,
    )
    test_model.fit(development_features, development_labels)
    test_predictions = test_model.predict(test_features)
    test_scores = test_model.predict_proba(test_features).max(axis=1)
    test_accepted = test_scores >= score_threshold
    class_names = sorted(set(labels))

    print(f"\nHeld-out grouped test images: {len(test_labels)}")
    print(f"Test accuracy: {accuracy_score(test_labels, test_predictions):.2%}")
    print(
        "Test macro-F1: "
        f"{f1_score(test_labels, test_predictions, average='macro', zero_division=0):.2%}"
    )
    print("\nPer-class test report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=class_names,
            zero_division=0,
        )
    )
    print("Confusion matrix labels:", class_names)
    print(confusion_matrix(test_labels, test_predictions, labels=class_names))
    print(f"Test accepted coverage: {np.mean(test_accepted):.2%}")
    if np.any(test_accepted):
        accepted_accuracy = accuracy_score(
            test_labels[test_accepted],
            test_predictions[test_accepted],
        )
        print(f"Test accuracy when accepted: {accepted_accuracy:.2%}")
    print(f"Test images returned as Unknown Object: {np.sum(~test_accepted)}")

    final_model = RandomForestClassifier(
        n_estimators=selected_tree_count,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(features, labels)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "image_size": IMAGE_SIZE,
            "name": f"Random Forest ({selected_tree_count} trees)",
            "score_threshold": score_threshold,
            "feature_version": "hog-grayscale-128-overlay-clean-v1",
        },
        MODEL_PATH,
    )
    print(f"\nSaved final model, trained on all images, to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
