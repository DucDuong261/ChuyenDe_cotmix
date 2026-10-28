"""
Prepare raw UCI-HAR (Kaggle copy of the official UCI HAR Dataset) for CoTMix.

Expected input layout:
  UCI HAR Dataset/
    train/
      Inertial Signals/
        body_acc_x_train.txt ...
      subject_train.txt
      y_train.txt
    test/
      Inertial Signals/
        body_acc_x_test.txt ...
      subject_test.txt
      y_test.txt

Output layout expected by this CoTMix repo:
  CoTMix/data/HAR/train_2.pt
  CoTMix/data/HAR/test_2.pt
  ...
  CoTMix/data/HAR/train_30.pt
  CoTMix/data/HAR/test_30.pt

Each .pt is a dict:
  samples: float32 tensor [N, 9, 128]
  labels:  int64 tensor [N], labels remapped from 1..6 to 0..5

Important protocol note:
The official UCI-HAR split separates people between train and test (subjects 1..21 vs 22..30).
CoTMix's provided HAR config expects both train_ID.pt and test_ID.pt for every domain ID,
so this script combines the official train/test records, groups windows by subject, then
splits each subject's own windows into 80% train / 20% test. This is a subject-specific
within-domain split, not the official UCI-HAR evaluation protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split


SIGNALS = [
    ("body_acc_x", "body_acc"),
    ("body_acc_y", "body_acc"),
    ("body_acc_z", "body_acc"),
    ("body_gyro_x", "body_gyro"),
    ("body_gyro_y", "body_gyro"),
    ("body_gyro_z", "body_gyro"),
    ("total_acc_x", "total_acc"),
    ("total_acc_y", "total_acc"),
    ("total_acc_z", "total_acc"),
]


def find_dataset_root(user_path: Path) -> Path:
    """Find the directory that contains train/ and test/."""
    candidates = [
        user_path,
        user_path / "UCI HAR Dataset",
        user_path / "UCI_HAR_Dataset",
        user_path / "UCI HAR",
    ]
    for candidate in candidates:
        if (
            (candidate / "train" / "Inertial Signals").is_dir()
            and (candidate / "test" / "Inertial Signals").is_dir()
        ):
            return candidate

    # Search a few levels below the supplied directory (common Kaggle zip layout).
    if user_path.is_dir():
        for candidate in user_path.glob("**/train/Inertial Signals"):
            root = candidate.parent.parent
            if (root / "test" / "Inertial Signals").is_dir():
                return root

    raise FileNotFoundError(
        "Không tìm thấy thư mục train/Inertial Signals và test/Inertial Signals. "
        "Hãy truyền --input_dir tới thư mục UCI HAR Dataset hoặc thư mục cha của nó."
    )


def load_partition(root: Path, partition: str):
    signal_dir = root / partition / "Inertial Signals"
    suffix = "train" if partition == "train" else "test"

    channels = []
    for signal_name, _group in SIGNALS:
        path = signal_dir / f"{signal_name}_{suffix}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file tín hiệu: {path}")
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 128:
            raise ValueError(
                f"{path} có shape {values.shape}; mong đợi [N, 128]. "
                "Đây phải là bản UCI-HAR có thư mục Inertial Signals."
            )
        channels.append(values)

    # [N, 9, 128], consistent channel order: body_acc xyz, body_gyro xyz, total_acc xyz.
    x = np.stack(channels, axis=1).astype(np.float32, copy=False)

    subject_path = root / partition / f"subject_{suffix}.txt"
    labels_path = root / partition / f"y_{suffix}.txt"
    if not subject_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Thiếu {subject_path.name} hoặc {labels_path.name} trong {root / partition}"
        )

    subjects = np.loadtxt(subject_path, dtype=np.int64).reshape(-1)
    labels = np.loadtxt(labels_path, dtype=np.int64).reshape(-1)

    if not (len(x) == len(subjects) == len(labels)):
        raise ValueError(
            f"Số mẫu không khớp trong {partition}: X={len(x)}, "
            f"subject={len(subjects)}, y={len(labels)}"
        )

    # UCI-HAR labels are 1..6; CoTMix classifier expects class indices 0..5.
    labels = labels - 1
    if labels.min() < 0 or labels.max() > 5:
        raise ValueError("Nhãn UCI-HAR ngoài khoảng 1..6; kiểm tra y_train/y_test.txt.")

    return x, subjects, labels


def split_indices(indices, labels, test_size: float, seed: int):
    """Stratify where possible, but handle small per-subject subsets safely."""
    if len(indices) < 2:
        raise ValueError(
            "Một subject có ít hơn 2 cửa sổ, không thể chia train/test. "
            "Hãy kiểm tra dữ liệu đầu vào."
        )

    n_test = max(1, int(round(len(indices) * test_size)))
    n_test = min(n_test, len(indices) - 1)

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
            stratify=labels[indices],
        )
    except ValueError:
        # Fallback for rare classes with too few examples for stratification.
        train_idx, test_idx = train_test_split(
            indices,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.asarray(train_idx), np.asarray(test_idx)


def save_subject(output_dir: Path, subject_id: int, x, subjects, y, test_size: float, seed: int):
    indices = np.flatnonzero(subjects == subject_id)
    if len(indices) == 0:
        print(f"[WARN] Subject {subject_id}: không có mẫu, bỏ qua.")
        return False

    train_idx, test_idx = split_indices(indices, y, test_size, seed)

    for split_name, selected in (("train", train_idx), ("test", test_idx)):
        payload = {
            "samples": torch.from_numpy(np.ascontiguousarray(x[selected])).float(),
            "labels": torch.from_numpy(np.ascontiguousarray(y[selected])).long(),
        }
        path = output_dir / f"{split_name}_{subject_id}.pt"
        torch.save(payload, path)
        print(
            f"[OK] {path.name}: samples={tuple(payload['samples'].shape)}, "
            f"labels={tuple(payload['labels'].shape)}"
        )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw UCI-HAR Inertial Signals into CoTMix per-subject .pt files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to UCI HAR Dataset, or its parent folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/HAR"),
        help="Output directory (default: data/HAR relative to current working directory).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Within-subject test fraction (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help='Subject IDs, e.g. "2,6,7,9,11,12,13,16,18,23", or "all".',
    )
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test_size phải nằm trong khoảng (0, 1).")

    root = find_dataset_root(args.input_dir)
    print(f"Dataset root: {root}")

    x_train, s_train, y_train = load_partition(root, "train")
    x_test, s_test, y_test = load_partition(root, "test")

    # Combine rows from the official partitions, then make per-subject splits.
    x = np.concatenate([x_train, x_test], axis=0)
    subjects_all = np.concatenate([s_train, s_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    if args.subjects.strip().lower() == "all":
        requested_subjects = sorted(np.unique(subjects_all).tolist())
    else:
        try:
            requested_subjects = sorted(
                {int(item.strip()) for item in args.subjects.split(",") if item.strip()}
            )
        except ValueError:
            parser.error('--subjects phải là "all" hoặc danh sách số, ví dụ "2,11".')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for subject_id in requested_subjects:
        written += int(save_subject(
            args.output_dir, subject_id, x, subjects_all, y, args.test_size, args.seed
        ))

    print(f"\nHoàn tất: tạo file cho {written} subject tại {args.output_dir.resolve()}")
    print("Tensor shape dự kiến: samples=[N, 9, 128], labels=[N].")


if __name__ == "__main__":
    main()
