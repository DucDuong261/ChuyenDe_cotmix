import os
import sys
import csv
import io
import json
import html
import time
import random
import tempfile
from datetime import datetime

import gradio as gr
import numpy as np
import torch

# -------------------- Cấu hình mô hình --------------------
torch.set_num_threads(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_PT_PATH = os.path.join(BASE_DIR, "cotmix_uci_har.pt")
SAMPLES_PATH = os.path.join(BASE_DIR, "samples", "uci_har_test_samples.npz")
EXPORT_DIR = os.path.join(tempfile.gettempdir(), "phonepose_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
device = torch.device("cpu")

SEQ_LEN = 128
SAMPLING_HZ = 50

# Thứ tự kênh giống prepare_uci_har_cotmix.py (đầu vào mô hình: 1 x 9 x 128)
CHANNELS = [
    {"key": "body_acc_x", "desc": "Gia tốc cơ thể trục X", "unit": "g", "color": "#3B82F6"},
    {"key": "body_acc_y", "desc": "Gia tốc cơ thể trục Y", "unit": "g", "color": "#14B8A6"},
    {"key": "body_acc_z", "desc": "Gia tốc cơ thể trục Z", "unit": "g", "color": "#8B5CF6"},
    {"key": "body_gyro_x", "desc": "Vận tốc góc trục X", "unit": "rad/s", "color": "#F59E0B"},
    {"key": "body_gyro_y", "desc": "Vận tốc góc trục Y", "unit": "rad/s", "color": "#EF4444"},
    {"key": "body_gyro_z", "desc": "Vận tốc góc trục Z", "unit": "rad/s", "color": "#EC4899"},
    {"key": "total_acc_x", "desc": "Gia tốc tổng trục X (có trọng lực)", "unit": "g", "color": "#0EA5E9"},
    {"key": "total_acc_y", "desc": "Gia tốc tổng trục Y (có trọng lực)", "unit": "g", "color": "#22C55E"},
    {"key": "total_acc_z", "desc": "Gia tốc tổng trục Z (có trọng lực)", "unit": "g", "color": "#64748B"},
]
CHANNEL_KEYS = [c["key"] for c in CHANNELS]
DEFAULT_CHANNELS = CHANNEL_KEYS[:3]

# Thứ tự lớp giống configs/data_model_configs.py: walk, upstairs, downstairs, sit, stand, lie
CLASSES = [
    {"key": "walk", "vi": "Đi bộ", "en": "Walking", "color": "#7B4FE0"},
    {"key": "upstairs", "vi": "Đi lên cầu thang", "en": "Walking upstairs", "color": "#22B8CF"},
    {"key": "downstairs", "vi": "Đi xuống cầu thang", "en": "Walking downstairs", "color": "#F06595"},
    {"key": "sit", "vi": "Ngồi", "en": "Sitting", "color": "#4C6EF5"},
    {"key": "stand", "vi": "Đứng", "en": "Standing", "color": "#4DABF7"},
    {"key": "lie", "vi": "Nằm", "en": "Laying", "color": "#9775FA"},
]

PIPELINE_STEPS = ["Đầu vào", "Tiền xử lý", "Cắt cửa sổ", "Mã hóa", "Suy luận", "Kết quả"]


def load_model():
    try:
        m = torch.jit.load(MODEL_PT_PATH, map_location=device)
        m.eval()
        return m, None
    except Exception as e:
        return None, str(e)


def load_samples():
    if not os.path.exists(SAMPLES_PATH):
        return None
    d = np.load(SAMPLES_PATH)
    return {"x": d["x"].astype(np.float32), "y": d["y"], "subject": d["subject"], "index": d["index"]}


model, model_error = load_model()
SAMPLES = load_samples()


# -------------------- Biểu tượng SVG --------------------
def icon(paths, size=18, stroke=2.0, cls=""):
    return (
        f'<svg class="ico {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{paths}</svg>'
    )


ICON_PULSE = '<path d="M3 12h4l3-8 4 16 3-8h4"/>'
ICON_HISTORY = '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>'
ICON_FILE = '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M8 15l2-2 2 2 3-3"/>'
ICON_INFO = '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>'
ICON_CHECK = '<path d="M5 12l5 5L20 7"/>'
ICON_ALERT = '<path d="M12 3l9.5 16.5H2.5z"/><path d="M12 10v4"/><path d="M12 17h.01"/>'
ICON_CHART = '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>'
ICON_DB = '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'

POSTURE_ICONS = {
    "stand": '<circle cx="12" cy="4" r="2"/><path d="M12 7v7"/><path d="M8 10l4-2 4 2"/><path d="M12 14l-2.5 7"/><path d="M12 14l2.5 7"/>',
    "sit": '<circle cx="9" cy="4" r="2"/><path d="M9 7v7h6v6"/><path d="M9 10l4 2"/><path d="M5 12v8"/><path d="M5 16h4"/>',
    "lie": '<circle cx="4.5" cy="11" r="2"/><path d="M7 12.5h12"/><path d="M11 12.5l-2-3"/><path d="M19 12.5l2 2"/><path d="M2 17h20"/>',
    "walk": '<circle cx="13" cy="4" r="2"/><path d="M12.5 7l-1.5 6"/><path d="M11 13l-3 7"/><path d="M11 13l4 3v4"/><path d="M12 9l-3 2"/><path d="M12 9l3 1.5"/>',
    "upstairs": '<path d="M3 20h4v-4h4v-4h4v-4h4"/><path d="M14 3h6v6"/><path d="M20 3l-7 7"/>',
    "downstairs": '<path d="M21 20h-4v-4h-4v-4H9V8H5"/><path d="M4 9V3h6"/><path d="M4 3l7 7"/>',
}

LOGO_SVG = """
<svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
  <path d="M4 13V7a3 3 0 0 1 3-3h6M27 4h6a3 3 0 0 1 3 3v6M4 27v6a3 3 0 0 0 3 3h6M27 36h6a3 3 0 0 0 3-3v-6" stroke="#1D6FE8" stroke-width="2.6" stroke-linecap="round"/>
  <circle cx="16.5" cy="11.5" r="2.8" stroke="#1D6FE8" stroke-width="2.2"/>
  <path d="M11 21l4-4.5 4.5 3 4-3.5M16 17.5L14.5 25l-3 6M16.5 23.5l4 3 1.5 5" stroke="#1D6FE8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M23 22h2.5l1.5-3.5 2 6 1.8-3.5H33" stroke="#12B886" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


# -------------------- Đọc dữ liệu --------------------
def _to_float(token):
    try:
        v = float(str(token).strip())
    except (TypeError, ValueError):
        return None
    return v


def array_from_matrix(rows, header):
    """Chuyển bảng số (đã bỏ header) thành mảng 9 x 128, kèm ghi chú cho người dùng."""
    notes = []
    if header:
        cols = [h.strip().lower() for h in header]
        known = [k for k in CHANNEL_KEYS if k in cols]
        if known:
            missing = [k for k in CHANNEL_KEYS if k not in cols]
            if missing:
                raise ValueError("Thiếu cột kênh: " + ", ".join(missing))
            idx = [cols.index(k) for k in CHANNEL_KEYS]
            rows = [[r[i] for i in idx] for r in rows if len(r) > max(idx)]

    widths = {len(r) for r in rows}
    if len(widths) == 1 and rows:
        mat = np.asarray(rows, dtype=np.float32)
        n_rows, n_cols = mat.shape
        if n_cols == 9 and n_rows >= SEQ_LEN:
            if n_rows > SEQ_LEN:
                notes.append(f"Tệp có {n_rows} mốc thời gian, chỉ dùng {SEQ_LEN} mốc đầu tiên.")
            return mat[:SEQ_LEN].T.copy(), n_rows, notes
        if n_rows == 9 and n_cols >= SEQ_LEN:
            if n_cols > SEQ_LEN:
                notes.append(f"Mỗi kênh có {n_cols} giá trị, chỉ dùng {SEQ_LEN} giá trị đầu tiên.")
            return mat[:, :SEQ_LEN].copy(), n_cols, notes

    flat = np.asarray([v for r in rows for v in r], dtype=np.float32)
    need = 9 * SEQ_LEN
    if flat.size < need:
        raise ValueError(
            f"Tệp chỉ có {flat.size:,} giá trị số, cần tối thiểu {need:,} "
            f"({SEQ_LEN} mốc × 9 kênh). Xem tab Chuẩn bị dữ liệu để biết định dạng."
        )
    if flat.size > need:
        notes.append(f"Tệp có {flat.size:,} giá trị, chỉ dùng {need:,} giá trị đầu tiên (9 kênh × {SEQ_LEN}).")
    notes.append("Dữ liệu được đọc theo thứ tự từng kênh nối tiếp nhau.")
    return flat[:need].reshape(9, SEQ_LEN), SEQ_LEN, notes


def parse_signal_file(path):
    ext = os.path.splitext(path)[1].lower()
    header = None
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        rows = np.atleast_2d(arr.squeeze()).tolist()
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and any(k in obj for k in CHANNEL_KEYS):
            missing = [k for k in CHANNEL_KEYS if k not in obj]
            if missing:
                raise ValueError("Thiếu kênh trong JSON: " + ", ".join(missing))
            rows = [list(map(float, obj[k])) for k in CHANNEL_KEYS]
        else:
            if isinstance(obj, dict):
                obj = obj.get("data", obj.get("values", []))
            arr = np.asarray(obj, dtype=np.float32)
            rows = np.atleast_2d(arr.squeeze()).tolist()
    else:
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Tệp rỗng.")
        delim = "," if "," in lines[0] else (";" if ";" in lines[0] else ("\t" if "\t" in lines[0] else None))
        parsed = []
        for ln in lines:
            cells = list(csv.reader([ln], delimiter=delim))[0] if delim else ln.split()
            parsed.append([c.strip() for c in cells if c.strip() != ""])
        if parsed and any(_to_float(c) is None for c in parsed[0]):
            header, parsed = parsed[0], parsed[1:]
        rows = []
        for r in parsed:
            vals = [_to_float(c) for c in r]
            if any(v is None for v in vals):
                raise ValueError(f"Dòng có giá trị không phải số: {', '.join(r[:4])}…")
            rows.append(vals)
    if not rows:
        raise ValueError("Không tìm thấy giá trị số trong tệp.")
    x, n_steps, notes = array_from_matrix(rows, header)
    if not np.all(np.isfinite(x)):
        raise ValueError("Dữ liệu có giá trị rỗng hoặc NaN/Inf.")
    return x, n_steps, notes


def write_signal_csv(x, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CHANNEL_KEYS)
        for t in range(x.shape[1]):
            w.writerow([f"{v:.6f}" for v in x[:, t]])
    return path


def sample_record(i):
    label = int(SAMPLES["y"][i])
    subject = int(SAMPLES["subject"][i])
    return {
        "x": SAMPLES["x"][i],
        "name": f"uci_har_{CLASSES[label]['key']}_s{subject:02d}_{int(SAMPLES['index'][i])}.csv",
        "source": "sample",
        "steps": SEQ_LEN,
        "notes": [],
        "truth": label,
        "subject": subject,
        "sample_idx": i,
    }


TEMPLATE_CSV = None
if SAMPLES is not None:
    _stand = int(np.where(SAMPLES["y"] == 4)[0][0])
    TEMPLATE_CSV = write_signal_csv(SAMPLES["x"][_stand], os.path.join(EXPORT_DIR, "phonepose_du_lieu_mau.csv"))


# -------------------- Thành phần giao diện (HTML/SVG) --------------------
def esc(s):
    return html.escape(str(s))


def status_pill_html():
    if model is not None:
        return '<div class="pp-status ok" role="status"><span class="dot"></span>Mô hình sẵn sàng</div>'
    return (
        f'<div class="pp-status err" role="status" title="{esc(model_error)}">'
        '<span class="dot"></span>Mô hình chưa kết nối</div>'
    )


def brand_html():
    return f'<div class="pp-brand">{LOGO_SVG}<span><b>Phone</b><i>Pose</i></span></div>'


def file_card_html(cur):
    if cur is None:
        return f"""
        <div class="pp-file empty">
          <div class="pp-file-ico">{icon(ICON_FILE, 22)}</div>
          <div class="pp-file-meta">
            <div class="pp-file-name">Chưa có tệp tín hiệu</div>
            <div class="pp-file-sub">Chọn dữ liệu mẫu hoặc tải lên tệp .csv / .txt / .json / .npy</div>
          </div>
        </div>
        <div class="pp-hint">{icon(ICON_INFO, 14)}<span>Gợi ý: mỗi hàng là một mốc thời gian, mỗi cột là một kênh tín hiệu, ví dụ: body_acc_x, body_acc_y, body_gyro_z…</span></div>
        """
    if cur["source"] == "sample":
        sub = f"{SEQ_LEN} mốc • 9 kênh • Người tham gia #{cur['subject']} • Nhãn thật: {CLASSES[cur['truth']]['vi']}"
        tag = '<span class="pp-chip blue">Mẫu UCI-HAR</span>'
    else:
        sub = f"{cur['steps']} mốc trong tệp • 9 kênh tín hiệu"
        tag = '<span class="pp-chip green">Tệp của bạn</span>'
    notes = "".join(
        f'<div class="pp-hint warn">{icon(ICON_ALERT, 14)}<span>{esc(n)}</span></div>' for n in cur["notes"]
    )
    return f"""
    <div class="pp-file">
      <div class="pp-file-ico">{icon(ICON_FILE, 22)}</div>
      <div class="pp-file-meta">
        <div class="pp-file-name" title="{esc(cur['name'])}">{esc(cur['name'])} {tag}</div>
        <div class="pp-file-sub">{esc(sub)}</div>
      </div>
    </div>
    {notes or f'<div class="pp-hint">{icon(ICON_INFO, 14)}<span>Gợi ý: mỗi hàng là một mốc thời gian, mỗi cột là một kênh tín hiệu, ví dụ: body_acc_x, body_acc_y, body_gyro_z…</span></div>'}
    """


def _nice_ticks(lo, hi, n=5):
    if hi - lo < 1e-6:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / (n - 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = np.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 6))
        v += step
    return ticks


def waveform_html(cur, channels):
    head = """
    <div class="pp-card-head">
      <div><h2>Dạng sóng cảm biến</h2><p>Quan sát sự thay đổi của tín hiệu theo thời gian.</p></div>
      {badge}
    </div>"""
    if cur is None:
        return head.format(badge="") + f"""
        <div class="pp-empty tall">{icon(ICON_PULSE, 34, 1.6)}
          <b>Chưa có tín hiệu để hiển thị</b>
          <span>Nhấn <em>Dùng dữ liệu mẫu</em> hoặc <em>Chọn tệp</em> ở trên để bắt đầu.</span>
        </div>"""
    badge = (
        f'<span class="pp-chip blue"><span class="dot"></span>Dữ liệu thật UCI-HAR · {esc(CLASSES[cur["truth"]]["vi"])}</span>'
        if cur["source"] == "sample"
        else '<span class="pp-chip green"><span class="dot"></span>Tệp tải lên</span>'
    )
    channels = [c for c in CHANNEL_KEYS if c in (channels or [])]
    if not channels:
        return head.format(badge=badge) + f"""
        <div class="pp-empty tall">{icon(ICON_PULSE, 34, 1.6)}<b>Chưa chọn kênh nào</b>
        <span>Chọn ít nhất một kênh ở bên dưới để xem dạng sóng.</span></div>"""

    x = np.asarray(cur["x"])
    sel = [CHANNEL_KEYS.index(c) for c in channels]
    data = x[sel]
    W, H = 920, 250
    L, R, T, B = 52, 14, 12, 34
    pw, ph = W - L - R, H - T - B
    lo, hi = float(data.min()), float(data.max())
    pad = (hi - lo) * 0.08 or 0.5
    ticks = _nice_ticks(lo - pad, hi + pad)
    y0, y1 = ticks[0], ticks[-1]

    def px(t):
        return L + pw * t / (SEQ_LEN - 1)

    def py(v):
        return T + ph * (1 - (v - y0) / (y1 - y0))

    decimals = 0 if (ticks[1] - ticks[0]) >= 1 else (1 if (ticks[1] - ticks[0]) >= 0.1 else 2)
    grid = ""
    for tv in ticks:
        yy = py(tv)
        grid += f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid{" zero" if abs(tv) < 1e-9 else ""}"/>'
        grid += f'<text x="{L - 8}" y="{yy + 4:.1f}" class="ytick">{tv:.{decimals}f}</text>'
    for t in (0, 32, 64, 96, 127):
        grid += f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{T}" y2="{T + ph}" class="grid v"/>'
        grid += f'<text x="{px(t):.1f}" y="{H - 14}" class="xtick">{t}</text>'

    lines = ""
    for ch_i, row in zip(sel, data):
        pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in enumerate(row))
        lines += f'<polyline points="{pts}" fill="none" stroke="{CHANNELS[ch_i]["color"]}" stroke-width="2" stroke-linejoin="round"/>'

    # Vùng hover: mỗi mốc thời gian một cột trong suốt, tooltip hiển thị giá trị chính xác
    step_w = pw / (SEQ_LEN - 1)
    hover = ""
    for t in range(SEQ_LEN):
        vals = " · ".join(f"{CHANNEL_KEYS[i]}: {x[i, t]:+.3f}" for i in sel)
        hover += (
            f'<rect x="{px(t) - step_w / 2:.1f}" y="{T}" width="{step_w:.1f}" height="{ph}" class="hov">'
            f"<title>Mốc {t} ({t / SAMPLING_HZ:.2f}s) — {vals}</title></rect>"
        )
    legend = "".join(
        f'<span><i style="background:{CHANNELS[i]["color"]}"></i>{CHANNEL_KEYS[i]}</span>' for i in sel
    )
    aria = f"Biểu đồ đường {len(sel)} kênh tín hiệu trong {SEQ_LEN} mốc thời gian, giá trị từ {lo:.2f} đến {hi:.2f}."
    return head.format(badge=badge) + f"""
    <div class="pp-chart">
      <div class="pp-chart-top"><b>Biểu đồ tín hiệu</b><div class="pp-legend">{legend}</div></div>
      <svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" class="pp-wave">
        {grid}{lines}{hover}
        <text x="{L + pw / 2:.0f}" y="{H - 1}" class="xlabel">Thời gian (mẫu, {SAMPLING_HZ} Hz · {SEQ_LEN / SAMPLING_HZ:.2f} giây)</text>
      </svg>
    </div>
    """


def metrics_html(cur):
    if cur is None:
        steps, chans, status = "—", "—", '<span class="muted">Chưa có dữ liệu</span>'
    else:
        steps, chans = SEQ_LEN, 9
        status = f'<span class="ok">{icon(ICON_CHECK, 15, 2.6)} Sẵn sàng phân loại</span>'
    return f"""
    <div class="pp-metrics">
      <div><span>Số mốc thời gian</span><b>{steps}</b></div>
      <div><span>Kênh sử dụng</span><b>{chans}</b></div>
      <div><span>Trạng thái dữ liệu</span><b>{status}</b></div>
    </div>
    """


def pipeline_html(stage):
    """stage 0: chưa có dữ liệu, 1: đã nạp & tiền xử lý xong, 2: đã phân loại."""
    done_upto = {0: 0, 1: 3, 2: 6}[stage]
    items = ""
    for i, name in enumerate(PIPELINE_STEPS, start=1):
        state = "done" if i <= done_upto else ("active" if i == done_upto + 1 else "")
        mark = icon(ICON_CHECK, 15, 2.8) if state == "done" else str(i)
        items += f'<li class="{state}"><span class="n">{mark}</span><span class="t">{name}</span></li>'
    return f"""
    <div class="pp-card-head slim">
      <h3>Quy trình phân loại</h3>
      <span class="pp-sub-note">Tín hiệu → đặc trưng → dự đoán</span>
    </div>
    <ol class="pp-steps" aria-label="Tiến trình: đã hoàn thành {done_upto}/6 bước">{items}</ol>
    """


def confidence_level(p):
    if p >= 80:
        return "high", "Độ tin cậy cao"
    if p >= 50:
        return "mid", "Độ tin cậy trung bình"
    return "low", "Độ tin cậy thấp"


def result_html(pred):
    if pred is None:
        return f"""
        <div class="pp-result empty">
          <div class="pp-empty">{icon(POSTURE_ICONS['stand'], 34, 1.6)}
            <b>Chưa có kết quả phân loại</b>
            <span>Nạp tín hiệu rồi nhấn <em>Phân loại tư thế</em>. Kết quả sẽ hiển thị tại đây.</span>
          </div>
        </div>"""
    probs = np.asarray(pred["probs"])
    order = np.argsort(-probs)
    best, second = int(order[0]), int(order[1])
    c = CLASSES[best]
    conf = probs[best] * 100
    lvl, lvl_txt = confidence_level(conf)
    truth_html = ""
    if pred.get("truth") is not None:
        t = CLASSES[pred["truth"]]
        if pred["truth"] == best:
            truth_html = f'<div class="pp-truth ok">{icon(ICON_CHECK, 15, 2.6)} Khớp nhãn thật UCI-HAR: <b>{t["vi"]}</b></div>'
        else:
            truth_html = f'<div class="pp-truth bad">{icon(ICON_ALERT, 15)} Khác nhãn thật UCI-HAR: <b>{t["vi"]}</b></div>'
    tip = ""
    if lvl == "low":
        tip = '<div class="pp-hint warn">' + icon(ICON_ALERT, 14) + "<span>Độ tin cậy thấp — hãy kiểm tra lại tệp, thứ tự kênh và đơn vị đo (g, rad/s).</span></div>"
    return f"""
    <div class="pp-result">
      <div class="pp-result-ico" style="--c:{c['color']}">{icon(POSTURE_ICONS[c['key']], 40, 1.8)}</div>
      <div class="pp-result-main">
        <div class="pp-result-k">Tư thế được nhận diện</div>
        <div class="pp-result-name">{c['vi']}</div>
        <div class="pp-result-sub">Nhãn mô hình: {c['en']} · Khả năng thứ hai: {CLASSES[second]['vi']} ({probs[second] * 100:.1f}%)</div>
        <div class="pp-result-conf"><b>{conf:.1f}%</b><span>độ tin cậy</span><span class="pp-level {lvl}">{lvl_txt}</span></div>
        {truth_html}
      </div>
      <div class="pp-result-side">
        <span class="pp-chip">Thời gian suy luận: {pred['ms']:.0f} ms</span>
        <span class="pp-chip">{esc(pred['time'])}</span>
      </div>
    </div>
    {tip}
    """


def empty_chart_html(title, sub=""):
    return f"""
    <div class="pp-card-head"><div><h2>{title}</h2>{f'<p>{sub}</p>' if sub else ''}</div><span class="pp-chip">6 nhóm tư thế</span></div>
    <div class="pp-empty tall">{icon(ICON_CHART, 34, 1.6)}
      <b>Chưa có dữ liệu dự đoán</b>
      <span>Nhấn <em>Phân loại tư thế</em> để xem xác suất của 6 nhóm hoạt động.</span>
    </div>"""


def distribution_html(pred):
    if pred is None:
        return empty_chart_html("Phân bố hoạt động")
    probs = np.asarray(pred["probs"]) * 100
    order = np.argsort(-probs)
    r, cx, cy, sw = 70, 90, 90, 26
    circ = 2 * np.pi * r
    segs, offset = "", 0.0
    for i in order:
        length = circ * probs[i] / 100
        if length < 0.01:
            continue
        gap = 2 if length > 4 else 0
        segs += (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{CLASSES[i]["color"]}" stroke-width="{sw}" '
            f'stroke-dasharray="{max(length - gap, 0.01):.2f} {circ:.2f}" stroke-dashoffset="{-offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"><title>{CLASSES[i]["vi"]}: {probs[i]:.1f}%</title></circle>'
        )
        offset += length
    best = int(order[0])
    legend = "".join(
        f'<li><i style="background:{CLASSES[i]["color"]}"></i><span>{CLASSES[i]["vi"]}</span><b>{probs[i]:.1f}%</b></li>'
        for i in order
    )
    bars = "".join(
        f'<div class="pp-bar"><span>{CLASSES[i]["vi"]}</span>'
        f'<div class="track"><div class="fill" style="width:{max(probs[i], 0.6):.1f}%;background:{CLASSES[i]["color"]}"></div></div>'
        f"<b>{probs[i]:.1f}%</b></div>"
        for i in order
    )
    aria = "; ".join(f"{CLASSES[i]['vi']} {probs[i]:.1f}%" for i in order)
    return f"""
    <div class="pp-card-head"><div><h2>Phân bố hoạt động</h2></div><span class="pp-chip">6 nhóm tư thế</span></div>
    <div class="pp-donut-wrap">
      <svg viewBox="0 0 180 180" class="pp-donut" role="img" aria-label="Biểu đồ tròn xác suất: {esc(aria)}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#EEF4FD" stroke-width="{sw}"/>
        {segs}
        <text x="{cx}" y="{cy + 2}" class="big">{probs[best]:.0f}%</text>
        <text x="{cx}" y="{cy + 22}" class="small">{CLASSES[best]['vi']}</text>
      </svg>
      <ul class="pp-legend-list">{legend}</ul>
    </div>
    <div class="pp-bars">{bars}</div>
    <div class="pp-foot"><span>Nguồn: {esc(pred['name'])}</span></div>
    """


def ranking_html(pred):
    if pred is None:
        return empty_chart_html("Xếp hạng nhóm hoạt động", "So sánh điểm số của các nhóm dự đoán.")
    probs = np.asarray(pred["probs"]) * 100
    order = np.argsort(-probs)
    W, H = 420, 250
    L, R, T, B = 34, 8, 22, 62
    pw, ph = W - L - R, H - T - B
    slot = pw / 6
    bw = slot * 0.56
    svg = ""
    for tv in (0, 25, 50, 75, 100):
        yy = T + ph * (1 - tv / 100)
        svg += f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
        svg += f'<text x="{L - 6}" y="{yy + 4:.1f}" class="ytick">{tv}</text>'
    for rank, i in enumerate(order, start=1):
        v = probs[i]
        bh = max(ph * v / 100, 1.5)
        x0 = L + slot * (rank - 1) + (slot - bw) / 2
        y0 = T + ph - bh
        name = CLASSES[i]["vi"]
        words = name.split(" ")
        l1, l2 = (" ".join(words[:2]), " ".join(words[2:])) if len(words) > 2 else (name, "")
        cxm = x0 + bw / 2
        svg += (
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{CLASSES[i]["color"]}">'
            f"<title>#{rank} {name}: {v:.1f}%</title></rect>"
            f'<text x="{cxm:.1f}" y="{y0 - 6:.1f}" class="val">{v:.1f}</text>'
            f'<text x="{cxm:.1f}" y="{T + ph + 16}" class="cat">{l1}</text>'
            + (f'<text x="{cxm:.1f}" y="{T + ph + 29}" class="cat">{l2}</text>' if l2 else "")
            + f'<text x="{cxm:.1f}" y="{T + ph + 46}" class="rank">#{rank}</text>'
        )
    aria = ", ".join(f"hạng {r} {CLASSES[i]['vi']} {probs[i]:.1f}%" for r, i in enumerate(order, 1))
    return f"""
    <div class="pp-card-head"><div><h2>Xếp hạng nhóm hoạt động</h2><p>So sánh điểm số (%) của các nhóm dự đoán.</p></div><span class="pp-chip blue">TOP 6</span></div>
    <svg viewBox="0 0 {W} {H}" class="pp-rank" role="img" aria-label="{esc(aria)}">{svg}</svg>
    <div class="pp-foot right"><span>Thang đo 0 – 100%</span></div>
    """


def history_html(history):
    if not history:
        return f"""
        <div class="pp-empty tall">{icon(ICON_HISTORY, 34, 1.6)}
          <b>Chưa có lượt phân loại nào trong phiên này</b>
          <span>Kết quả mỗi lần nhấn <em>Phân loại tư thế</em> sẽ được lưu tại đây để so sánh.</span>
        </div>"""
    total = len(history)
    avg = sum(h["conf"] for h in history) / total
    labeled = [h for h in history if h["truth"] is not None]
    match = sum(1 for h in labeled if h["truth"] == h["pred"])
    acc_txt = f"{match}/{len(labeled)}" if labeled else "—"
    rows = ""
    for n, h in enumerate(reversed(history), start=1):
        c = CLASSES[h["pred"]]
        if h["truth"] is None:
            verdict = '<span class="muted">Không có nhãn</span>'
        elif h["truth"] == h["pred"]:
            verdict = f'<span class="pp-tag ok">{icon(ICON_CHECK, 13, 2.6)} Khớp</span>'
        else:
            verdict = f'<span class="pp-tag bad">{icon(ICON_ALERT, 13)} Lệch</span>'
        truth = CLASSES[h["truth"]]["vi"] if h["truth"] is not None else "—"
        rows += f"""
        <tr>
          <td>{total - n + 1}</td><td>{esc(h['time'])}</td>
          <td class="name" title="{esc(h['name'])}">{esc(h['name'])}</td>
          <td><span class="pp-dot" style="background:{c['color']}"></span><b>{c['vi']}</b></td>
          <td class="num">{h['conf']:.1f}%</td><td>{truth}</td><td>{verdict}</td>
        </tr>"""
    return f"""
    <div class="pp-metrics hist">
      <div><span>Tổng lượt phân loại</span><b>{total}</b></div>
      <div><span>Độ tin cậy trung bình</span><b>{avg:.1f}%</b></div>
      <div><span>Khớp nhãn thật (mẫu UCI-HAR)</span><b>{acc_txt}</b></div>
    </div>
    <div class="pp-table-wrap">
      <table class="pp-table">
        <thead><tr><th>#</th><th>Thời gian</th><th>Nguồn dữ liệu</th><th>Dự đoán</th><th class="num">Độ tin cậy</th><th>Nhãn thật</th><th>Đánh giá</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def data_guide_html():
    ch_rows = "".join(
        f'<tr><td><span class="pp-dot" style="background:{c["color"]}"></span><code>{c["key"]}</code></td><td>{c["desc"]}</td><td>{c["unit"]}</td><td class="num">{i + 1}</td></tr>'
        for i, c in enumerate(CHANNELS)
    )
    return f"""
    <div class="pp-guide">
      <div class="pp-guide-cards">
        <div><span class="n">1</span><b>Kích thước cửa sổ</b><p>{SEQ_LEN} mốc thời gian × 9 kênh, lấy mẫu {SAMPLING_HZ} Hz (≈ {SEQ_LEN / SAMPLING_HZ:.2f} giây chuyển động).</p></div>
        <div><span class="n">2</span><b>Bố cục bảng</b><p>Mỗi hàng là một mốc thời gian, mỗi cột là một kênh. Nên có dòng tiêu đề với tên kênh như bảng bên dưới.</p></div>
        <div><span class="n">3</span><b>Định dạng hỗ trợ</b><p><code>.csv</code>, <code>.txt</code> (dấu phẩy / chấm phẩy / tab / khoảng trắng), <code>.json</code> (đối tượng theo tên kênh hoặc mảng), <code>.npy</code>.</p></div>
        <div><span class="n">4</span><b>Tự động xử lý</b><p>Tệp dài hơn {SEQ_LEN} mốc sẽ dùng {SEQ_LEN} mốc đầu. Bảng 9 × {SEQ_LEN} (kênh theo hàng) cũng được chấp nhận.</p></div>
      </div>
      <h3>Danh sách kênh tín hiệu (đúng thứ tự đầu vào mô hình)</h3>
      <div class="pp-table-wrap">
        <table class="pp-table">
          <thead><tr><th>Tên cột</th><th>Ý nghĩa</th><th>Đơn vị</th><th class="num">Thứ tự</th></tr></thead>
          <tbody>{ch_rows}</tbody>
        </table>
      </div>
      <h3>Ví dụ tệp CSV</h3>
      <pre class="pp-code">body_acc_x,body_acc_y,body_acc_z,body_gyro_x,body_gyro_y,body_gyro_z,total_acc_x,total_acc_y,total_acc_z
0.000181,0.010767,0.055561,0.030191,0.066014,0.022859,1.012817,-0.123217,0.102934
0.010139,0.006579,0.055125,0.043711,0.042699,0.010316,1.022833,-0.126876,0.105687
… (tổng cộng {SEQ_LEN} hàng dữ liệu)</pre>
    </div>
    """


# -------------------- Xử lý sự kiện --------------------
def render_data(cur, channels):
    stage = 0 if cur is None else 1
    return (
        cur,
        None,
        file_card_html(cur),
        waveform_html(cur, channels),
        metrics_html(cur),
        pipeline_html(stage),
        result_html(None),
        distribution_html(None),
        ranking_html(None),
        gr.DownloadButton(value=None, interactive=False),
    )


def on_use_sample(choice, cur, channels):
    if SAMPLES is None:
        raise gr.Error("Không tìm thấy bộ dữ liệu mẫu (samples/uci_har_test_samples.npz).")
    pool = list(range(len(SAMPLES["y"])))
    if choice and choice != "Ngẫu nhiên":
        cls = next(i for i, c in enumerate(CLASSES) if c["vi"] == choice)
        pool = [i for i in pool if int(SAMPLES["y"][i]) == cls]
    if cur is not None and cur.get("sample_idx") in pool and len(pool) > 1:
        pool.remove(cur["sample_idx"])
    rec = sample_record(random.choice(pool))
    gr.Info(f"Đã nạp mẫu UCI-HAR: {CLASSES[rec['truth']]['vi']} (người tham gia #{rec['subject']}).")
    return render_data(rec, channels)


def on_upload(file_obj, channels):
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", "")
    try:
        x, n_steps, notes = parse_signal_file(path)
    except Exception as exc:
        raise gr.Error(f"Không đọc được tệp: {exc}", duration=8)
    rec = {
        "x": x,
        "name": os.path.basename(path),
        "source": "file",
        "steps": n_steps,
        "notes": notes,
        "truth": None,
        "subject": None,
        "sample_idx": None,
    }
    gr.Info(f"Đã nạp tệp {rec['name']}.")
    return render_data(rec, channels)


def on_clear(channels):
    return render_data(None, channels)


def on_channels(cur, channels):
    return waveform_html(cur, channels)


def export_prediction_csv(pred):
    path = os.path.join(EXPORT_DIR, f"phonepose_ket_qua_{datetime.now():%Y%m%d_%H%M%S}.csv")
    probs = np.asarray(pred["probs"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["hang", "hoat_dong", "nhan_mo_hinh", "xac_suat_pct"])
        for rank, i in enumerate(np.argsort(-probs), start=1):
            w.writerow([rank, CLASSES[i]["vi"], CLASSES[i]["en"], f"{probs[i] * 100:.2f}"])
    return path


def export_history_csv(history):
    path = os.path.join(EXPORT_DIR, f"phonepose_lich_su_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["thoi_gian", "nguon_du_lieu", "du_doan", "do_tin_cay_pct", "nhan_that", "khop"])
        for h in history:
            truth = CLASSES[h["truth"]]["vi"] if h["truth"] is not None else ""
            match = "" if h["truth"] is None else ("co" if h["truth"] == h["pred"] else "khong")
            w.writerow([h["time"], h["name"], CLASSES[h["pred"]]["vi"], f"{h['conf']:.2f}", truth, match])
    return path


def on_classify(cur, history):
    no_change = (gr.update(),) * 7
    if cur is None:
        gr.Warning("Chưa có tín hiệu. Hãy chọn dữ liệu mẫu hoặc tải tệp lên trước.")
        return no_change
    if model is None:
        raise gr.Error(f"Không tải được mô hình cotmix_uci_har.pt: {model_error}")
    tensor = torch.from_numpy(np.asarray(cur["x"], dtype=np.float32)).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    ms = (time.perf_counter() - start) * 1000
    pred = {
        "probs": probs.tolist(),
        "ms": ms,
        "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        "name": cur["name"],
        "truth": cur["truth"],
    }
    best = int(np.argmax(probs))
    history = list(history or []) + [
        {"time": pred["time"], "name": cur["name"], "pred": best, "conf": float(probs[best]) * 100, "truth": cur["truth"]}
    ]
    return (
        pred,
        pipeline_html(2),
        result_html(pred),
        distribution_html(pred),
        ranking_html(pred),
        gr.DownloadButton(value=export_prediction_csv(pred), interactive=True),
        history,
    )


def on_history_change(history):
    return (
        history_html(history),
        gr.DownloadButton(value=export_history_csv(history) if history else None, interactive=bool(history)),
        gr.Button(interactive=bool(history)),
    )


def show_view(name):
    views = ["analyze", "history", "prepare"]
    return [gr.Column(visible=(v == name)) for v in views] + [
        gr.Button(elem_classes=["pp-nav", "active"] if v == name else ["pp-nav"]) for v in views
    ]


# -------------------- CSS --------------------
CSS = r"""
:root, .dark {
  --pp-bg: #F4F7FC;
  --pp-surface: #FFFFFF;
  --pp-surface-2: #F6F9FE;
  --pp-border: #E3EAF5;
  --pp-text: #0F2747;
  --pp-text-2: #4A5F7E;
  --pp-muted: #5F7290;
  --pp-primary: #1D6FE8;
  --pp-primary-soft: #E8F1FE;
  --pp-success: #0E9F6E;
  --pp-success-soft: #E3F7EF;
  --pp-danger: #D9363E;
  --pp-danger-soft: #FDECEC;
  --pp-warn: #B26A00;
  --pp-warn-soft: #FFF5E0;
  --pp-radius: 16px;
  --pp-shadow: 0 1px 2px rgba(15, 39, 71, .04), 0 4px 16px rgba(15, 39, 71, .05);
}
body, gradio-app, .gradio-container, .main, .wrap, .contain {
  background: var(--pp-bg) !important;
  color: var(--pp-text) !important;
}
.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 0 20px 32px !important;
  font-family: "Be Vietnam Pro", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
footer { display: none !important; }
.gradio-container .prose, .gradio-container .html-container { padding: 0 !important; }
.ico { flex: none; vertical-align: middle; }
.muted { color: var(--pp-muted); font-weight: 500; }
em { font-style: normal; font-weight: 600; color: var(--pp-primary); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

/* ---------- Top bar ---------- */
#pp-topbar {
  position: sticky; top: 0; z-index: 50;
  background: rgba(255, 255, 255, .92) !important;
  backdrop-filter: blur(8px);
  border: 1px solid var(--pp-border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--pp-radius) var(--pp-radius) !important;
  padding: 6px 18px !important;
  margin-bottom: 20px !important;
  align-items: center !important;
  gap: 4px !important;
  flex-wrap: wrap !important;
  box-shadow: var(--pp-shadow);
}
#pp-topbar > * { min-width: 0 !important; }
#pp-topbar .block, #pp-topbar .html-container { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.pp-brand { display: flex; align-items: center; gap: 10px; font-size: 21px; letter-spacing: -.4px; }
.pp-brand b { color: var(--pp-text); font-weight: 800; }
.pp-brand i { color: var(--pp-primary); font-style: normal; font-weight: 800; }
#pp-brand-col { flex: 1 1 auto !important; }
.gradio-container button { white-space: nowrap; }
#pp-topbar > button, #pp-topbar > .html-container, #pp-topbar > div:last-child { flex: 0 0 auto !important; width: auto !important; }
button.pp-nav {
  flex: 0 0 auto !important; width: auto !important;
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: var(--pp-text-2) !important; font-weight: 500 !important; font-size: 14.5px !important;
  border-radius: 0 !important; padding: 18px 16px !important; min-height: 60px;
  border-bottom: 3px solid transparent !important; cursor: pointer;
  transition: color .15s ease, border-color .15s ease;
}
button.pp-nav:hover { color: var(--pp-primary) !important; }
button.pp-nav.active { color: var(--pp-primary) !important; font-weight: 700 !important; border-bottom-color: var(--pp-primary) !important; }
button.pp-nav:focus-visible, .gradio-container button:focus-visible { outline: 3px solid #9CC3FF !important; outline-offset: 2px; }
.pp-status { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; white-space: nowrap; margin-left: 12px; }
.pp-status .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 22%, transparent); }
.pp-status.ok { background: var(--pp-success-soft); color: var(--pp-success); }
.pp-status.err { background: var(--pp-danger-soft); color: var(--pp-danger); }

/* ---------- Cards ---------- */
.pp-card {
  background: var(--pp-surface) !important;
  border: 1px solid var(--pp-border) !important;
  border-radius: var(--pp-radius) !important;
  box-shadow: var(--pp-shadow) !important;
  padding: 20px !important;
  gap: 14px !important;
}
.pp-card .block, .pp-card .form, .pp-card .html-container, .pp-plain, .pp-plain .block {
  background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
.pp-sub-card { background: var(--pp-surface-2) !important; border: 1px solid var(--pp-border) !important; border-radius: 12px !important; padding: 14px 16px !important; }
.pp-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.pp-card-head.slim { margin-bottom: 8px; align-items: center; }
.pp-card-head h2, .pp-title { font-size: 18px; font-weight: 700; color: var(--pp-text); margin: 0; line-height: 1.35; }
.pp-card-head h3, .pp-guide h3 { font-size: 15px; font-weight: 700; color: var(--pp-text); margin: 0; }
.pp-card-head p { font-size: 13.5px; color: var(--pp-muted); margin: 4px 0 0; }
.pp-sub-note { font-size: 12.5px; color: var(--pp-muted); }
.pp-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; background: var(--pp-surface-2); border: 1px solid var(--pp-border); color: var(--pp-text-2); white-space: nowrap; }
.pp-chip .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pp-chip.blue { background: var(--pp-primary-soft); border-color: #CFE0FB; color: var(--pp-primary); }
.pp-chip.green { background: var(--pp-success-soft); border-color: #BDEBD7; color: var(--pp-success); }

/* ---------- Hero ---------- */
#pp-hero { align-items: flex-end !important; gap: 12px !important; flex-wrap: wrap !important; }
#pp-hero .block, #pp-hero .html-container { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }
.pp-hero h1 { font-size: 28px; font-weight: 700; letter-spacing: -.5px; margin: 0 0 6px; color: var(--pp-text); line-height: 1.25; text-wrap: balance; }
.pp-hero p { margin: 0; color: var(--pp-muted); font-size: 14.5px; line-height: 1.55; max-width: 68ch; }

/* ---------- Buttons ---------- */
.gradio-container button.pp-btn-primary, .gradio-container .pp-btn-primary button {
  background: var(--pp-primary) !important; color: #fff !important; border: none !important;
  border-radius: 10px !important; font-weight: 600 !important; min-height: 44px; padding: 0 20px !important;
  box-shadow: 0 6px 16px rgba(29, 111, 232, .25) !important; cursor: pointer; transition: background .15s ease, transform .1s ease;
}
.gradio-container button.pp-btn-primary:hover { background: #155CC7 !important; }
.gradio-container button.pp-btn-primary:active { transform: scale(.98); }
.gradio-container button.pp-btn-ghost, .gradio-container .pp-btn-ghost button, .gradio-container a.pp-btn-ghost {
  background: var(--pp-surface) !important; color: var(--pp-text) !important;
  border: 1px solid var(--pp-border) !important; border-radius: 10px !important; font-weight: 600 !important;
  min-height: 44px; box-shadow: none !important; cursor: pointer; transition: border-color .15s ease, color .15s ease;
}
.gradio-container button.pp-btn-ghost:hover { border-color: var(--pp-primary) !important; color: var(--pp-primary) !important; }
.gradio-container button.pp-btn-soft {
  background: var(--pp-primary-soft) !important; color: var(--pp-primary) !important; border: 1px solid #CFE0FB !important;
  border-radius: 999px !important; font-weight: 600 !important; min-height: 44px; box-shadow: none !important; cursor: pointer;
}
.gradio-container button.pp-btn-soft:hover { background: #D9E8FD !important; }
.gradio-container button.pp-btn-danger { background: transparent !important; color: var(--pp-danger) !important; border: 1px solid #F3C9CB !important; border-radius: 10px !important; min-height: 40px; box-shadow: none !important; }
.gradio-container button.pp-btn-danger:hover { background: var(--pp-danger-soft) !important; }
.gradio-container button:disabled { opacity: .45 !important; cursor: not-allowed !important; box-shadow: none !important; }

/* ---------- Input card ---------- */
#pp-input-card { flex-wrap: wrap !important; }
.pp-file { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border: 1.5px dashed #B9D0F2; border-radius: 12px; background: #F9FBFF; }
.pp-file.empty { border-color: var(--pp-border); }
.pp-file-ico { width: 44px; height: 44px; border-radius: 10px; display: grid; place-items: center; background: var(--pp-primary-soft); color: var(--pp-primary); flex: none; }
.pp-file.empty .pp-file-ico { background: var(--pp-surface-2); color: var(--pp-muted); }
.pp-file-meta { min-width: 0; }
.pp-file-name { font-weight: 700; font-size: 14.5px; color: var(--pp-text); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; overflow-wrap: anywhere; }
.pp-file-sub { font-size: 13px; color: var(--pp-muted); margin-top: 3px; }
.pp-hint { display: flex; gap: 8px; align-items: flex-start; font-size: 12.5px; color: var(--pp-muted); margin-top: 10px; line-height: 1.5; }
.pp-hint .ico { margin-top: 2px; }
.pp-hint.warn { color: var(--pp-warn); background: var(--pp-warn-soft); padding: 8px 12px; border-radius: 8px; }
.pp-card [data-testid="block-info"] { background: transparent !important; color: var(--pp-text-2) !important; font-weight: 600 !important; font-size: 13px !important; padding: 0 !important; border: none !important; box-shadow: none !important; }
.pp-field-label { font-size: 13px; font-weight: 600; color: var(--pp-text-2); margin: 2px 0 2px; }
.pp-card .wrap-inner, .pp-card input { font-size: 14px !important; }

/* ---------- Waveform ---------- */
.pp-chart { border: 1px solid var(--pp-border); border-radius: 12px; padding: 12px 14px 4px; }
.pp-chart-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 13.5px; color: var(--pp-text); }
.pp-legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12.5px; color: var(--pp-text-2); }
.pp-legend span { display: inline-flex; align-items: center; gap: 6px; }
.pp-legend i { width: 18px; height: 4px; border-radius: 2px; display: inline-block; }
.pp-wave { width: 100%; height: auto; display: block; }
.pp-wave .grid, .pp-rank .grid { stroke: #EDF2FA; stroke-width: 1; }
.pp-wave .grid.zero { stroke: #CBD8EC; }
.pp-wave .grid.v { stroke-dasharray: 3 4; }
.pp-wave .ytick, .pp-rank .ytick { font-size: 11px; fill: var(--pp-muted); text-anchor: end; }
.pp-wave .xtick { font-size: 11px; fill: var(--pp-muted); text-anchor: middle; }
.pp-wave .xlabel { font-size: 11.5px; fill: var(--pp-muted); text-anchor: middle; }
.pp-wave .hov { fill: transparent; cursor: crosshair; }
.pp-wave .hov:hover { fill: rgba(29, 111, 232, .10); }
#pp-channels { margin-top: 2px; }
#pp-channels .wrap { gap: 6px !important; }
#pp-channels label { border-radius: 999px !important; padding: 4px 12px !important; font-size: 12.5px !important; border: 1px solid var(--pp-border) !important; background: var(--pp-surface) !important; min-height: 32px; }
#pp-channels label.selected { background: var(--pp-primary-soft) !important; border-color: #A9C8F5 !important; color: var(--pp-primary) !important; }
.pp-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.pp-metrics > div { background: var(--pp-surface-2); border: 1px solid var(--pp-border); border-radius: 10px; padding: 10px 14px; }
.pp-metrics span { display: block; font-size: 12.5px; color: var(--pp-muted); }
.pp-metrics b { display: block; font-size: 17px; color: var(--pp-primary); margin-top: 4px; font-variant-numeric: tabular-nums; }
.pp-metrics b .ok { color: var(--pp-success); display: inline-flex; align-items: center; gap: 5px; font-size: 15px; }
.pp-metrics b .muted { font-size: 15px; }

/* ---------- Pipeline ---------- */
.pp-steps { list-style: none; margin: 4px 0 0; padding: 0; display: flex; justify-content: space-between; position: relative; }
.pp-steps li { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; position: relative; font-size: 12.5px; color: var(--pp-muted); }
.pp-steps li:not(:last-child)::after { content: ""; position: absolute; top: 17px; left: calc(50% + 22px); right: calc(-50% + 22px); height: 2px; background: #DCE6F5; border-radius: 2px; }
.pp-steps li.done:not(:last-child)::after { background: var(--pp-primary); }
.pp-steps .n { width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 14px; border: 2px solid #C9D7EC; background: #fff; color: var(--pp-muted); transition: all .25s ease; }
.pp-steps li.done .n { background: var(--pp-primary); border-color: var(--pp-primary); color: #fff; }
.pp-steps li.active .n { border-color: var(--pp-primary); color: var(--pp-primary); box-shadow: 0 0 0 4px var(--pp-primary-soft); }
.pp-steps li.done .t, .pp-steps li.active .t { color: var(--pp-text); font-weight: 600; }

/* ---------- Result ---------- */
.pp-result { display: flex; gap: 18px; align-items: flex-start; border: 1px solid var(--pp-border); border-radius: 12px; padding: 16px; flex-wrap: wrap; }
.pp-result.empty { display: block; padding: 0; }
.pp-result-ico { width: 72px; height: 72px; border-radius: 16px; display: grid; place-items: center; color: var(--c); background: color-mix(in srgb, var(--c) 14%, white); flex: none; }
.pp-result-main { flex: 1; min-width: 220px; }
.pp-result-k { font-size: 13px; color: var(--pp-muted); }
.pp-result-name { font-size: 26px; font-weight: 800; color: var(--pp-text); line-height: 1.25; }
.pp-result-sub { font-size: 13px; color: var(--pp-muted); margin-top: 2px; }
.pp-result-conf { display: flex; align-items: baseline; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.pp-result-conf b { font-size: 26px; color: var(--pp-primary); font-variant-numeric: tabular-nums; }
.pp-result-conf span { font-size: 13px; color: var(--pp-muted); }
.pp-level { padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 12px !important; }
.pp-level.high { background: var(--pp-success-soft); color: var(--pp-success) !important; }
.pp-level.mid { background: var(--pp-warn-soft); color: var(--pp-warn) !important; }
.pp-level.low { background: var(--pp-danger-soft); color: var(--pp-danger) !important; }
.pp-truth { display: inline-flex; align-items: center; gap: 6px; margin-top: 10px; font-size: 13px; padding: 6px 12px; border-radius: 8px; }
.pp-truth.ok { background: var(--pp-success-soft); color: var(--pp-success); }
.pp-truth.bad { background: var(--pp-danger-soft); color: var(--pp-danger); }
.pp-result-side { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }

/* ---------- Empty states ---------- */
.pp-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 6px; padding: 26px 16px; color: var(--pp-muted); border: 1.5px dashed var(--pp-border); border-radius: 12px; background: var(--pp-surface-2); }
.pp-empty.tall { min-height: 220px; }
.pp-empty b { color: var(--pp-text); font-size: 14.5px; }
.pp-empty span { font-size: 13px; max-width: 40ch; line-height: 1.5; }
.pp-empty .ico { color: #9DB4D6; margin-bottom: 4px; }

/* ---------- Distribution ---------- */
.pp-donut-wrap { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.pp-donut { width: 180px; height: 180px; flex: none; margin: 0 auto; }
.pp-donut circle { transition: stroke-dasharray .5s ease; }
.pp-donut .big { font-size: 26px; font-weight: 800; fill: var(--pp-text); text-anchor: middle; }
.pp-donut .small { font-size: 12.5px; fill: var(--pp-muted); text-anchor: middle; }
.pp-legend-list { list-style: none; padding: 0; margin: 0; flex: 1; min-width: 180px; display: grid; gap: 9px; }
.pp-legend-list li { display: grid; grid-template-columns: 12px 1fr auto; align-items: center; gap: 10px; font-size: 13.5px; color: var(--pp-text-2); }
.pp-legend-list i { width: 11px; height: 11px; border-radius: 50%; }
.pp-legend-list b { color: var(--pp-primary); font-variant-numeric: tabular-nums; }
.pp-bars { display: grid; gap: 8px; margin-top: 18px; }
.pp-bar { display: grid; grid-template-columns: 128px 1fr 48px; align-items: center; gap: 10px; font-size: 12.5px; color: var(--pp-text-2); }
.pp-bar .track { height: 8px; background: #EEF3FA; border-radius: 99px; overflow: hidden; }
.pp-bar .fill { height: 100%; border-radius: 99px; transition: width .5s ease; }
.pp-bar b { text-align: right; font-weight: 600; color: var(--pp-text-2); font-variant-numeric: tabular-nums; }
.pp-foot { display: flex; justify-content: space-between; gap: 8px; margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--pp-border); font-size: 12px; color: var(--pp-muted); overflow-wrap: anywhere; }
.pp-foot.right { justify-content: flex-end; }
.pp-rank { width: 100%; height: auto; display: block; }
.pp-rank .val { font-size: 12px; font-weight: 700; fill: var(--pp-text); text-anchor: middle; }
.pp-rank .cat { font-size: 11.5px; fill: var(--pp-text-2); text-anchor: middle; }
.pp-rank .rank { font-size: 11px; font-weight: 700; fill: var(--pp-primary); text-anchor: middle; }
.pp-rank rect { transition: opacity .15s ease; }
.pp-rank rect:hover { opacity: .8; }

/* ---------- History / guide tables ---------- */
.pp-table-wrap { overflow-x: auto; border: 1px solid var(--pp-border); border-radius: 12px; margin-top: 14px; }
.pp-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.pp-table th { text-align: left; font-size: 12.5px; font-weight: 600; color: var(--pp-muted); background: var(--pp-surface-2); padding: 10px 14px; border-bottom: 1px solid var(--pp-border); white-space: nowrap; }
.pp-table td { padding: 10px 14px; border-bottom: 1px solid #EFF3F9; color: var(--pp-text-2); vertical-align: middle; }
.pp-table th, .pp-table td { border-left: none !important; border-right: none !important; border-top: none !important; }
.pp-table tr:last-child td { border-bottom: none; }
.pp-table tbody tr:hover td { background: #FAFCFF; }
.pp-table td b { color: var(--pp-text); }
.pp-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.pp-table td.name { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pp-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.pp-tag { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 999px; }
.pp-tag.ok { background: var(--pp-success-soft); color: var(--pp-success); }
.pp-tag.bad { background: var(--pp-danger-soft); color: var(--pp-danger); }
.pp-metrics.hist { margin-bottom: 0; }
.pp-guide h3 { margin: 22px 0 0; }
.pp-guide-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.pp-guide-cards > div { border: 1px solid var(--pp-border); border-radius: 12px; padding: 14px; background: var(--pp-surface-2); }
.pp-guide-cards .n { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: var(--pp-primary); color: #fff; font-size: 13px; font-weight: 700; }
.pp-guide-cards b { display: block; margin: 8px 0 4px; color: var(--pp-text); font-size: 14.5px; }
.pp-guide-cards p { margin: 0; font-size: 13px; color: var(--pp-text-2); line-height: 1.55; }
.pp-guide code, .pp-table code { background: var(--pp-primary-soft); color: #164FA8; padding: 1px 6px; border-radius: 5px; font-size: 12.5px; }
.pp-code { background: #0F2747; color: #DCE7F7; border-radius: 12px; padding: 14px 16px; font-size: 12.5px; line-height: 1.6; overflow-x: auto; margin-top: 10px; white-space: pre; }

/* ---------- Responsive ---------- */
@media (max-width: 1100px) {
  .pp-guide-cards { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px 24px !important; }
  .pp-hero h1 { font-size: 23px; }
  .pp-metrics { grid-template-columns: 1fr; }
  .pp-guide-cards { grid-template-columns: 1fr; }
  .pp-steps .t { font-size: 11px; text-align: center; }
  .pp-steps .n { width: 30px; height: 30px; font-size: 12.5px; }
  .pp-steps li:not(:last-child)::after { top: 14px; left: calc(50% + 18px); right: calc(-50% + 18px); }
  .pp-bar { grid-template-columns: 104px 1fr 44px; }
  button.pp-nav { padding: 12px 10px !important; min-height: 48px; font-size: 13.5px !important; }
  .pp-result-side { align-items: flex-start; }
}
"""

FORCE_LIGHT_JS = """
() => {
  document.body.classList.remove('dark');
  document.querySelector('gradio-app')?.classList.remove('dark');
}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=["Be Vietnam Pro", "system-ui", "sans-serif"],
)

# Gradio 6 chuyển css/theme/js/head từ Blocks sang launch()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
UI_KWARGS = dict(css=CSS, theme=THEME, js=FORCE_LIGHT_JS, head=HEAD)

INITIAL = sample_record(int(np.where(SAMPLES["y"] == 4)[0][1])) if SAMPLES is not None else None
SAMPLE_CHOICES = ["Ngẫu nhiên"] + [c["vi"] for c in CLASSES]
CHANNEL_CHOICES = [(c["key"], c["key"]) for c in CHANNELS]

with gr.Blocks(title="PhonePose · Nhận diện tư thế từ cảm biến", **({} if GRADIO_6 else UI_KWARGS)) as demo:
    current_state = gr.State(INITIAL)
    pred_state = gr.State(None)
    history_state = gr.State([])

    # -------------------- Thanh điều hướng --------------------
    with gr.Row(elem_id="pp-topbar", equal_height=True):
        with gr.Column(elem_id="pp-brand-col", scale=1, min_width=160):
            gr.HTML(brand_html())
        nav_analyze = gr.Button("Phân tích tín hiệu", elem_classes=["pp-nav", "active"], scale=0, min_width=0)
        nav_history = gr.Button("Lịch sử phân loại", elem_classes=["pp-nav"], scale=0, min_width=0)
        nav_prepare = gr.Button("Chuẩn bị dữ liệu", elem_classes=["pp-nav"], scale=0, min_width=0)
        gr.HTML(status_pill_html())

    # -------------------- Trang Phân tích tín hiệu --------------------
    with gr.Column(visible=True) as view_analyze:
        with gr.Row(equal_height=False):
            # Cột trái
            with gr.Column(scale=13, min_width=560):
                with gr.Row(elem_id="pp-hero"):
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML(
                            '<div class="pp-hero"><h1>Nhận diện tư thế từ tín hiệu cảm biến</h1>'
                            "<p>Tải dữ liệu chuyển động từ điện thoại, quan sát dạng sóng và phân loại sáu nhóm "
                            "hoạt động bằng mô hình CoTMix huấn luyện trên UCI-HAR.</p></div>"
                        )
                    btn_template = gr.DownloadButton(
                        "Tải dữ liệu mẫu",
                        value=TEMPLATE_CSV,
                        elem_classes=["pp-btn-ghost"],
                        scale=0,
                        min_width=150,
                    )
                    btn_classify_top = gr.Button("Phân loại tư thế", elem_classes=["pp-btn-primary"], scale=0, min_width=170)

                # Tệp tín hiệu đầu vào
                with gr.Row(elem_classes=["pp-card"], elem_id="pp-input-card", equal_height=False):
                    with gr.Column(scale=4, min_width=240):
                        gr.HTML('<h2 class="pp-title">Tệp tín hiệu đầu vào</h2>')
                        sample_choice = gr.Dropdown(
                            SAMPLE_CHOICES,
                            value="Ngẫu nhiên",
                            label="Hoạt động của dữ liệu mẫu",
                            info="Lấy cửa sổ thật từ tập kiểm thử UCI-HAR",
                        )
                        with gr.Row():
                            btn_sample = gr.Button("Dùng dữ liệu mẫu", elem_classes=["pp-btn-soft"], min_width=120)
                            btn_upload = gr.UploadButton(
                                "Chọn tệp",
                                file_types=[".csv", ".txt", ".json", ".npy"],
                                file_count="single",
                                elem_classes=["pp-btn-ghost"],
                                min_width=110,
                            )
                    with gr.Column(scale=7, min_width=300):
                        file_card = gr.HTML(file_card_html(INITIAL))
                        btn_clear = gr.Button("Xóa dữ liệu hiện tại", elem_classes=["pp-btn-danger"], size="sm")

                # Dạng sóng + quy trình + kết quả
                with gr.Column(elem_classes=["pp-card"]):
                    wave_html = gr.HTML(waveform_html(INITIAL, DEFAULT_CHANNELS))
                    channel_pick = gr.CheckboxGroup(
                        CHANNEL_CHOICES,
                        value=DEFAULT_CHANNELS,
                        label="Kênh hiển thị",
                        info="Rê chuột lên biểu đồ để xem giá trị chính xác tại từng mốc",
                        elem_id="pp-channels",
                    )
                    metrics = gr.HTML(metrics_html(INITIAL))
                    with gr.Column(elem_classes=["pp-sub-card"]):
                        pipeline = gr.HTML(pipeline_html(1 if INITIAL is not None else 0))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=240):
                            gr.HTML(
                                '<div class="pp-card-head" style="margin:0"><div><h2>Kết quả phân loại</h2>'
                                "<p>Dự đoán thực tế từ mô hình cotmix_uci_har.pt.</p></div></div>"
                            )
                        btn_classify = gr.Button("Phân loại tư thế", elem_classes=["pp-btn-soft"], scale=0, min_width=170)
                    result = gr.HTML(result_html(None))

            # Cột phải
            with gr.Column(scale=7, min_width=340):
                with gr.Column(elem_classes=["pp-card"]):
                    dist = gr.HTML(distribution_html(None))
                    btn_export = gr.DownloadButton(
                        "Xuất kết quả CSV", value=None, interactive=False, elem_classes=["pp-btn-ghost"], size="sm"
                    )
                with gr.Column(elem_classes=["pp-card"]):
                    rank = gr.HTML(ranking_html(None))

    # -------------------- Trang Lịch sử phân loại --------------------
    with gr.Column(visible=False) as view_history:
        with gr.Column(elem_classes=["pp-card"]):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML(
                        '<div class="pp-card-head" style="margin:0"><div><h2>Lịch sử phân loại</h2>'
                        "<p>Các lượt phân loại trong phiên làm việc hiện tại (không lưu lên máy chủ).</p></div></div>"
                    )
                btn_hist_export = gr.DownloadButton(
                    "Xuất lịch sử CSV", value=None, interactive=False, elem_classes=["pp-btn-ghost"], scale=0, min_width=160
                )
                btn_hist_clear = gr.Button(
                    "Xóa lịch sử", interactive=False, elem_classes=["pp-btn-danger"], scale=0, min_width=120
                )
            hist_html = gr.HTML(history_html([]))

    # -------------------- Trang Chuẩn bị dữ liệu --------------------
    with gr.Column(visible=False) as view_prepare:
        with gr.Column(elem_classes=["pp-card"]):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML(
                        '<div class="pp-card-head" style="margin:0"><div><h2>Chuẩn bị dữ liệu</h2>'
                        "<p>Định dạng tệp tín hiệu để mô hình đọc được chính xác.</p></div></div>"
                    )
                gr.DownloadButton(
                    "Tải tệp CSV mẫu", value=TEMPLATE_CSV, elem_classes=["pp-btn-primary"], scale=0, min_width=170
                )
            gr.HTML(data_guide_html())

    # -------------------- Sự kiện --------------------
    data_outputs = [current_state, pred_state, file_card, wave_html, metrics, pipeline, result, dist, rank, btn_export]
    classify_outputs = [pred_state, pipeline, result, dist, rank, btn_export, history_state]

    btn_sample.click(on_use_sample, [sample_choice, current_state, channel_pick], data_outputs)
    btn_upload.upload(on_upload, [btn_upload, channel_pick], data_outputs)
    btn_clear.click(on_clear, [channel_pick], data_outputs)
    channel_pick.change(on_channels, [current_state, channel_pick], wave_html, show_progress="hidden")
    for b in (btn_classify, btn_classify_top):
        b.click(on_classify, [current_state, history_state], classify_outputs, show_progress="minimal")

    history_state.change(on_history_change, history_state, [hist_html, btn_hist_export, btn_hist_clear])
    btn_hist_clear.click(lambda: [], None, history_state)

    nav_outputs = [view_analyze, view_history, view_prepare, nav_analyze, nav_history, nav_prepare]
    nav_analyze.click(lambda: show_view("analyze"), None, nav_outputs, show_progress="hidden")
    nav_history.click(lambda: show_view("history"), None, nav_outputs, show_progress="hidden")
    nav_prepare.click(lambda: show_view("prepare"), None, nav_outputs, show_progress="hidden")

interface = demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, **(UI_KWARGS if GRADIO_6 else {}))
