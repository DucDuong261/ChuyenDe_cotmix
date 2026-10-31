"""Smoke test cho app.py: nạp mô hình TorchScript, dữ liệu mẫu và chạy suy luận.

Chạy: pytest -q tests/
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402  (import app chỉ dựng giao diện, không launch server)


def test_model_loaded():
    assert app.model is not None, f"Không tải được mô hình: {app.model_error}"


def test_samples_loaded():
    assert app.SAMPLES is not None, "Thiếu samples/uci_har_test_samples.npz"
    x = app.SAMPLES["x"]
    assert x.ndim == 3 and x.shape[1:] == (9, app.SEQ_LEN)


def test_inference_shape_and_probs():
    x = torch.from_numpy(app.SAMPLES["x"][:1])
    with torch.no_grad():
        probs = torch.softmax(app.model(x), dim=1).numpy()
    assert probs.shape == (1, len(app.CLASSES))
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_accuracy_on_bundled_samples():
    # Mô hình hiện tại đạt ~83% trên 24 mẫu; chặn ở 70% để bắt lỗi hồi quy rõ rệt
    x = torch.from_numpy(app.SAMPLES["x"])
    with torch.no_grad():
        pred = app.model(x).argmax(dim=1).numpy()
    acc = float((pred == app.SAMPLES["y"]).mean())
    assert acc >= 0.70, f"Độ chính xác trên mẫu giảm còn {acc:.2%}"


@pytest.mark.parametrize("ext", [".csv", ".json"])
def test_parse_roundtrip(tmp_path, ext):
    x = app.SAMPLES["x"][0]
    path = tmp_path / f"signal{ext}"
    if ext == ".csv":
        app.write_signal_csv(x, str(path))
    else:
        import json
        path.write_text(json.dumps({k: x[i].tolist() for i, k in enumerate(app.CHANNEL_KEYS)}))
    arr, n_steps, _ = app.parse_signal_file(str(path))
    assert n_steps == app.SEQ_LEN
    assert np.allclose(arr, x, atol=1e-4)
