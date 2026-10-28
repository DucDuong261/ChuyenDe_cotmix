
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION — EDIT PATHS HERE IF NEEDED
# ============================================================

DATASET_ROOT = Path(
    r"D:\Bob\TaiLieuHoc\ChuyenDe\CoTMix_DINOv3_UCI_HAR\CoTMix\UCI-HAR Dataset"
)

OUTPUT_DIR = Path(
    r"D:\Bob\TaiLieuHoc\ChuyenDe\CoTMix_DINOv3_UCI_HAR\CoTMix\data\HAR"
)

TEST_SIZE = 0.2
RANDOM_SEED = 42
SUBJECTS = "all"  # "all" hoặc danh sách, ví dụ [2, 6, 11]

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def validate_paths():
    print(f"[INFO] Dataset root: {DATASET_ROOT}", flush=True)
    print(f"[INFO] Output dir: {OUTPUT_DIR}", flush=True)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dataset:\n{DATASET_ROOT}"
        )

    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE phải nằm trong khoảng (0, 1).")

    for part in ("train", "test"):
        signal_dir = DATASET_ROOT / part / "Inertial Signals"
        if not signal_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {signal_dir}"
            )

        required = [
            *(signal_dir / f"{name}_{part}.txt" for name in SIGNALS),
            DATASET_ROOT / part / f"subject_{part}.txt",
            DATASET_ROOT / part / f"y_{part}.txt",
        ]

        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Thiếu các file dataset:\n" + "\n".join(missing)
            )


def load_partition(part):
    signal_dir = DATASET_ROOT / part / "Inertial Signals"
    channels = []

    print(f"[INFO] Loading {part}...", flush=True)

    for name in SIGNALS:
        path = signal_dir / f"{name}_{part}.txt"
        values = np.loadtxt(path, dtype=np.float32)

        if values.ndim != 2 or values.shape[1] != 128:
            raise ValueError(
                f"{path} có shape {values.shape}; cần dạng [N, 128]."
            )

        channels.append(values)

    # [N, 9, 128]
    x = np.stack(channels, axis=1).astype(np.float32, copy=False)

    subjects = np.loadtxt(
        DATASET_ROOT / part / f"subject_{part}.txt",
        dtype=np.int64,
    ).reshape(-1)

    labels = np.loadtxt(
        DATASET_ROOT / part / f"y_{part}.txt",
        dtype=np.int64,
    ).reshape(-1)

    if not (len(x) == len(subjects) == len(labels)):
        raise ValueError(
            f"Số lượng mẫu không khớp trong {part}: "
            f"X={len(x)}, subjects={len(subjects)}, labels={len(labels)}"
        )

    # Đổi nhãn 1..6 thành 0..5
    labels = labels - 1

    if labels.min() < 0 or labels.max() > 5:
        raise ValueError(f"Nhãn không hợp lệ trong {part}; cần nhãn 1..6.")

    print(f"[INFO] Loaded {part}: {len(x)} windows", flush=True)
    return x, subjects, labels


def split_indices(indices, labels):
    if len(indices) < 2:
        raise ValueError("Subject có ít hơn 2 mẫu, không thể chia train/test.")

    n_test = max(1, round(len(indices) * TEST_SIZE))
    n_test = min(n_test, len(indices) - 1)

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=n_test,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=labels[indices],
        )
    except ValueError:
        # Dự phòng khi số mẫu của một số lớp quá ít.
        train_idx, test_idx = train_test_split(
            indices,
            test_size=n_test,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

    return train_idx, test_idx


def save_subject(subject_id, x, subjects, labels):
    indices = np.flatnonzero(subjects == subject_id)

    if len(indices) == 0:
        print(f"[WARN] Subject {subject_id} không có dữ liệu; bỏ qua.")
        return False

    train_idx, test_idx = split_indices(indices, labels)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, selected in (
        ("train", train_idx),
        ("test", test_idx),
    ):
        payload = {
            "samples": torch.from_numpy(
                np.ascontiguousarray(x[selected], dtype=np.float32)
            ),
            "labels": torch.from_numpy(
                np.ascontiguousarray(labels[selected], dtype=np.int64)
            ),
        }

        output_path = OUTPUT_DIR / f"{split_name}_{subject_id}.pt"
        torch.save(payload, output_path)

        print(
            f"[OK] {output_path.name}: "
            f"samples={tuple(payload['samples'].shape)}, "
            f"labels={tuple(payload['labels'].shape)}",
            flush=True,
        )

    return True


def main():
    print("[INFO] Starting UCI-HAR conversion...", flush=True)
    validate_paths()

    x_train, s_train, y_train = load_partition("train")
    x_test, s_test, y_test = load_partition("test")

    # Gộp dữ liệu train/test gốc rồi chia lại theo từng subject.
    x = np.concatenate([x_train, x_test], axis=0)
    subjects = np.concatenate([s_train, s_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)

    if isinstance(SUBJECTS, str) and SUBJECTS.lower() == "all":
        subject_ids = sorted(np.unique(subjects).tolist())
    else:
        subject_ids = sorted(set(int(s) for s in SUBJECTS))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for subject_id in subject_ids:
        if save_subject(subject_id, x, subjects, labels):
            written += 1

    print(
        f"\n[DONE] Đã tạo dữ liệu cho {written} subjects.",
        flush=True,
    )
    print(f"[DONE] Output directory: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
