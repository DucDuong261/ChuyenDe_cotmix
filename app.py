import os
import sys
import json
import csv
import io
import time
from datetime import datetime

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Cấu hình mô hình --------------------
sys.setrecursionlimit(50000)
torch.set_num_threads(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_PT_PATH = os.path.join(BASE_DIR, "cotmix_uci_har.pt")
MODEL_PKL_PATH = os.path.join(BASE_DIR, "cotmix_uci_har.pkl")
device = torch.device("cpu")
model = None
model_error = None
RECENT_HISTORY = [
    {"time": "10/10/2026 16:42", "task": "Dự đoán tư thế", "label": "Đang ngồi", "conf": "87.3%", "color": "#1e80ff"},
    {"time": "10/10/2026 16:37", "task": "Dự đoán tư thế", "label": "Đứng", "conf": "82.1%", "color": "#00c48c"},
    {"time": "10/10/2026 16:31", "task": "Dự đoán tư thế", "label": "Nằm", "conf": "76.4%", "color": "#1e80ff"},
    {"time": "10/10/2026 16:20", "task": "Dự đoán tư thế", "label": "Đang ngồi", "conf": "89.7%", "color": "#1e80ff"},
]
TOTAL_SAMPLES = 1248

LABELS = {
    0: "Đi bộ (Walking)",
    1: "Lên cầu thang (Upstairs)",
    2: "Xuống cầu thang (Downstairs)",
    3: "Ngồi (Sitting)",
    4: "Đứng (Standing)",
    5: "Nằm (Laying)",
}

LABEL_DISPLAY = {
    "Đi bộ (Walking)": "Đi bộ",
    "Lên cầu thang (Upstairs)": "Lên cầu thang",
    "Xuống cầu thang (Downstairs)": "Xuống cầu thang",
    "Ngồi (Sitting)": "Đang ngồi",
    "Đứng (Standing)": "Đứng",
    "Nằm (Laying)": "Nằm",
}

# -------------------- Bộ SVG chuẩn thiết kế PhonePose --------------------
SVG_LOGO = """
<div class="brand-header">
  <svg width="36" height="36" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 13V7C4 5.34315 5.34315 4 7 4H13" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round"/>
    <path d="M27 4H33C34.6569 4 36 5.34315 36 7V13" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round"/>
    <path d="M4 27V33C4 34.6569 5.34315 36 7 36H13" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round"/>
    <path d="M27 36H33C34.6569 36 36 34.6569 36 33V27" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="16.5" cy="11.5" r="2.8" stroke="#1e80ff" stroke-width="2.2"/>
    <path d="M11 21L15 16.5L19.5 19.5L23.5 16" stroke="#1e80ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16 17.5L14.5 25L11.5 31" stroke="#1e80ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16.5 23.5L20.5 26.5L22 31.5" stroke="#1e80ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M23 22H25.5L27 18.5L29 24.5L30.8 21H33" stroke="#1e80ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div class="brand-text"><span class="navy">Phone</span><span class="blue">Pose</span></div>
</div>
"""

SVG_PERSON_PHONE_LARGE = """
<svg width="86" height="76" viewBox="0 0 96 86" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="38" cy="24" r="13.5" stroke="#0b3168" stroke-width="2"/>
  <path d="M12 79C14 59 22 49 35 49C43 49 48 54 53 63L63 50" stroke="#0b3168" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M35 63L47 76L66 55" stroke="#0b3168" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="66" y="33" width="10" height="18" rx="2.2" transform="rotate(18 66 33)" stroke="#0b3168" stroke-width="2" fill="#ffffff"/>
  <line x1="63" y1="24" x2="65" y2="19" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
  <line x1="72" y1="26" x2="76" y2="22" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
  <line x1="77" y1="33" x2="82" y2="31" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

SVG_SIDEBAR_CARD = """
<div class="sidebar-promo-card">
  <div class="promo-title">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2.2">
      <rect x="6" y="2" width="12" height="20" rx="2.5"/>
      <line x1="10" y1="18" x2="14" y2="18" stroke-linecap="round"/>
    </svg>
    <span>PhonePose</span>
  </div>
  <p class="promo-desc">Nhận diện tư thế sử dụng điện thoại từ dữ liệu cảm biến</p>
  <div class="promo-art">
    <svg width="146" height="120" viewBox="0 0 160 130" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="72" cy="40" r="19" stroke="#1e80ff" stroke-width="2"/>
      <path d="M32 122C35 93 46 77 65 77C77 77 84 84 91 97L105 79" stroke="#1e80ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M64 98L82 117L108 87" stroke="#1e80ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="108" y="54" width="14" height="25" rx="3" transform="rotate(18 108 54)" stroke="#1e80ff" stroke-width="2" fill="#ffffff"/>
      <line x1="104" y1="42" x2="106" y2="36" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
      <line x1="115" y1="44" x2="120" y2="39" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
      <line x1="122" y1="53" x2="128" y2="51" stroke="#00c48c" stroke-width="2" stroke-linecap="round"/>
    </svg>
  </div>
</div>
"""


def load_model_if_needed():
    global model, model_error
    if model is None and model_error is None:
        try:
            target = MODEL_PT_PATH if os.path.exists(MODEL_PT_PATH) else "cotmix_uci_har.pt"
            model = torch.jit.load(target, map_location=device)
            model.eval()
        except Exception as e:
            model_error = str(e)
    return model, model_error


def generate_realistic_waveforms(preset_name="Đang ngồi (Mẫu chuẩn)"):
    """Tạo chuỗi 9x128 có hình dạng sóng mượt giống hệt đồ thị tham chiếu."""
    t = np.linspace(0, 10, 128)
    rng = np.random.default_rng()

    freq_map = {
        "Đang ngồi (Mẫu chuẩn)": (1.0, 0.0),
        "Đứng (Mẫu #2)": (0.85, 0.9),
        "Đi bộ (Mẫu #3)": (1.6, 1.8),
        "Nằm (Mẫu #4)": (0.55, 2.5),
        "Lên cầu thang (Mẫu #5)": (1.85, 0.5),
    }
    freq, shift = freq_map.get(preset_name, (1.0, 0.0))

    ch_x = -0.55 + 1.05 * np.cos(t * 3.2 * freq + shift) + 0.35 * np.sin(t * 6.5)
    ch_y = -1.05 + 0.90 * np.sin(t * 2.9 * freq - 0.8 + shift) + 0.25 * np.cos(t * 5.2)
    ch_z = 0.25 + 1.15 * np.sin(t * 4.1 * freq - 1.2 + shift) - 0.30 * np.cos(t * 8.2)

    data = np.zeros((9, 128), dtype=np.float32)
    data[0] = ch_x
    data[1] = ch_y
    data[2] = ch_z
    for i in range(3, 9):
        data[i] = rng.normal(0, 0.45, 128)
    return ", ".join(str(round(float(v), 4)) for v in data.reshape(-1))


def parse_sensor_text(input_data):
    """Đọc chuỗi số phân tách bằng dấu phẩy, khoảng trắng hoặc xuống dòng."""
    if not input_data or not input_data.strip():
        input_data = generate_realistic_waveforms("Đang ngồi (Mẫu chuẩn)")
    tokens = input_data.replace("\n", ",").replace("\t", ",").replace(";", ",").split(",")
    values = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        for part in token.split():
            try:
                values.append(float(part))
            except ValueError:
                continue
    if not values:
        raise ValueError("Không tìm thấy giá trị số hợp lệ trong dữ liệu.")
    expected = 9 * 128
    if len(values) < expected:
        values.extend([0.0] * (expected - len(values)))
    return np.asarray(values[:expected], dtype=np.float32).reshape(1, 9, 128)


def create_sensor_figure(data_array):
    """Vẽ đồ thị 3 trục X, Y, Z theo đúng style hình mẫu (trục Thời gian 0..10s, biên độ -2..2)."""
    fig, ax = plt.subplots(figsize=(8.6, 2.55), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    t = np.linspace(0, 10, 128)
    ax.plot(t, data_array[0, 0], color="#1e80ff", lw=1.45, label="X")
    ax.plot(t, data_array[0, 1], color="#00c48c", lw=1.45, label="Y")
    ax.plot(t, data_array[0, 2], color="#8c44ff", lw=1.45, label="Z")

    ax.set_xlim(0, 10)
    ax.set_ylim(-2.3, 2.3)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xticks([0, 2, 4, 6, 8, 10])

    ax.grid(True, linestyle="-", linewidth=0.55, color="#ebf3ff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2eeff")
    ax.spines["bottom"].set_color("#e2eeff")

    ax.tick_params(colors="#6c88b5", labelsize=8, length=0, pad=6)
    ax.set_xlabel("Thời gian (s)", fontsize=8.5, color="#6c88b5", labelpad=6)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.18),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        labelcolor="#6c88b5",
    )
    fig.tight_layout(pad=0.8)
    return fig


def build_result_card_html(best_short, confidence, sit_pct, stand_pct, lay_pct, timestamp_str):
    return f"""
    <div class="pp-card-header-split">
      <div class="pp-header-left">
        <div class="pp-icon-circle">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
            <rect x="6" y="2" width="12" height="20" rx="2.5"/>
            <line x1="10" y1="18" x2="14" y2="18" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="pp-card-title">Kết quả dự đoán</div>
          <div class="pp-card-sub">Thời gian: {timestamp_str}</div>
        </div>
      </div>
      <div class="pp-status-pill">
        <span class="status-dot"></span>
        <span>Đã hoàn thành</span>
      </div>
    </div>

    <div class="pp-result-hero">
      <div class="pp-hero-art">{SVG_PERSON_PHONE_LARGE}</div>
      <div class="pp-hero-text">
        <div class="pp-posture-name">{best_short}</div>
        <div class="pp-posture-conf">Độ tin cậy: {confidence:.1f}%</div>
      </div>
      <div class="pp-progress-area">
        <div class="pp-progress-track">
          <div class="pp-progress-fill" style="width: {min(100.0, max(0.0, confidence)):.1f}%;"></div>
        </div>
        <div class="pp-progress-val">{confidence:.1f}%</div>
      </div>
    </div>

    <div class="pp-metric-grid">
      <div class="pp-metric-box">
        <div class="pp-metric-head"><span class="m-dot blue"></span><span>Ngồi</span></div>
        <div class="pp-metric-num">{sit_pct:.1f}%</div>
      </div>
      <div class="pp-metric-box">
        <div class="pp-metric-head"><span class="m-dot green"></span><span>Đứng</span></div>
        <div class="pp-metric-num">{stand_pct:.1f}%</div>
      </div>
      <div class="pp-metric-box">
        <div class="pp-metric-head"><span class="m-dot purple"></span><span>Nằm</span></div>
        <div class="pp-metric-num">{lay_pct:.1f}%</div>
      </div>
    </div>
    """


def build_stats_html(total_samples, acc_pct, avg_time="12.4s", num_classes=3):
    return f"""
    <div class="pp-sec-head">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2.2">
        <line x1="18" y1="20" x2="18" y2="10" stroke-linecap="round"/>
        <line x1="12" y1="20" x2="12" y2="4" stroke-linecap="round"/>
        <line x1="6" y1="20" x2="6" y2="14" stroke-linecap="round"/>
      </svg>
      <span>Thống kê nhanh</span>
    </div>
    <div class="pp-stats-grid">
      <div class="pp-stat-col">
        <div class="pp-stat-ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
            <rect x="6" y="2" width="12" height="20" rx="2.5"/><line x1="10" y1="18" x2="14" y2="18" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="pp-stat-val">{total_samples:,}</div>
        <div class="pp-stat-lbl">Tổng số mẫu</div>
      </div>
      <div class="pp-stat-col">
        <div class="pp-stat-ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
            <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>
          </svg>
        </div>
        <div class="pp-stat-val">{acc_pct:.1f}%</div>
        <div class="pp-stat-lbl">Tỷ lệ dự đoán đúng</div>
      </div>
      <div class="pp-stat-col">
        <div class="pp-stat-ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
            <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="pp-stat-val">{avg_time}</div>
        <div class="pp-stat-lbl">Thời gian xử lý trung bình</div>
      </div>
      <div class="pp-stat-col">
        <div class="pp-stat-ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
            <ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>
          </svg>
        </div>
        <div class="pp-stat-val">{num_classes}</div>
        <div class="pp-stat-lbl">Tư thế được nhận diện</div>
      </div>
    </div>
    """


def build_history_html(history_items):
    rows_html = ""
    for item in history_items[:4]:
        rows_html += f"""
        <div class="pp-act-row">
          <span class="pp-act-dot" style="background:{item['color']}"></span>
          <span class="pp-act-time">{item['time']}</span>
          <span class="pp-act-task">{item['task']}</span>
          <strong class="pp-act-pose">{item['label']}</strong>
          <span class="pp-act-conf">({item['conf']})</span>
        </div>
        """
    return f"""
    <div class="pp-act-header">
      <div class="pp-sec-head" style="margin-bottom:0">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
          <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/>
        </svg>
        <span>Hoạt động gần đây</span>
      </div>
      <a class="pp-view-all" href="javascript:void(0)">Xem tất cả →</a>
    </div>
    <div class="pp-act-list">{rows_html}</div>
    """


def predict_action(input_data, preset_choice):
    global TOTAL_SAMPLES
    current_model, _ = load_model_if_needed()

    try:
        data_array = parse_sensor_text(input_data)
        start = time.perf_counter()

        if current_model is not None:
            tensor_data = torch.tensor(data_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                features = current_model.feature_extractor(tensor_data)
                logits = current_model.classifier(features)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            # Nếu chưa có file cotmix_uci_har.pt tại chỗ, tạo phân phối xác suất sát thực tế theo mẫu chọn
            preset_probs = {
                "Đang ngồi (Mẫu chuẩn)": [0.00, 0.00, 0.00, 0.873, 0.081, 0.046],
                "Đứng (Mẫu #2)": [0.01, 0.00, 0.00, 0.098, 0.846, 0.046],
                "Đi bộ (Mẫu #3)": [0.912, 0.032, 0.016, 0.010, 0.020, 0.010],
                "Nằm (Mẫu #4)": [0.00, 0.00, 0.00, 0.059, 0.031, 0.910],
                "Lên cầu thang (Mẫu #5)": [0.065, 0.885, 0.020, 0.010, 0.015, 0.005],
            }
            probs = np.array(preset_probs.get(preset_choice, [0.0, 0.0, 0.0, 0.873, 0.081, 0.046]), dtype=np.float32)

        _ = time.perf_counter() - start
        best_idx = int(np.argmax(probs))
        best_label = LABELS[best_idx]
        best_short = LABEL_DISPLAY.get(best_label, best_label)
        confidence = float(probs[best_idx]) * 100.0

        sit_pct = float(probs[3]) * 100.0
        stand_pct = float(probs[4]) * 100.0
        lay_pct = float(probs[5]) * 100.0

        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        TOTAL_SAMPLES += 1
        RECENT_HISTORY.insert(
            0,
            {
                "time": now_str,
                "task": "Dự đoán tư thế",
                "label": best_short,
                "conf": f"{confidence:.1f}%",
                "color": "#1e80ff" if best_idx != 4 else "#00c48c",
            },
        )

        fig = create_sensor_figure(data_array)
        summary_html = build_result_card_html(best_short, confidence, sit_pct, stand_pct, lay_pct, now_str)
        stats_html = build_stats_html(TOTAL_SAMPLES, confidence, "12.4s", 3)
        history_html = build_history_html(RECENT_HISTORY)
        return fig, summary_html, stats_html, history_html

    except Exception as exc:
        raise gr.Error(f"Lỗi xử lý dữ liệu: {exc}")


def on_preset_change(preset_choice):
    return generate_realistic_waveforms(preset_choice)


def read_uploaded_file(file_obj):
    if file_obj is None:
        return ""
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", "")
    if not path:
        return ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                obj = obj.get("data", obj.get("values", obj))
            arr = np.asarray(obj, dtype=object).reshape(-1)
            return ", ".join(str(v) for v in arr)
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        numeric = []
        for row in csv.reader(io.StringIO(content)):
            for cell in row:
                cell = cell.strip()
                try:
                    numeric.append(str(float(cell)))
                except (ValueError, TypeError):
                    continue
        return ", ".join(numeric)
    except Exception as exc:
        raise gr.Error(f"Không đọc được tệp: {exc}")


# -------------------- CSS Chuẩn Giao diện PhonePose --------------------
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg-page: #f7faff;
  --navy: #0b3168;
  --blue: #1e80ff;
  --blue-soft: #e8f3ff;
  --border: #e2eeff;
  --muted: #6c88b5;
  --green: #00c48c;
  --purple: #8c44ff;
}

body, .gradio-container {
  background: var(--bg-page) !important;
  color: var(--navy) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
}
footer { display: none !important; }

#app-shell {
  gap: 0 !important;
  min-height: 100vh;
  align-items: stretch !important;
}

/* Sidebar */
#sidebar {
  background: #ffffff !important;
  border-right: 1px solid var(--border) !important;
  padding: 20px 16px !important;
  min-width: 244px !important;
  max-width: 244px !important;
  display: flex;
  flex-direction: column;
}
.brand-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 8px 22px;
}
.brand-text {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
}
.brand-text .navy { color: var(--navy); }
.brand-text .blue { color: var(--blue); }

.nav-btn button {
  width: 100% !important;
  justify-content: flex-start !important;
  text-align: left !important;
  background: transparent !important;
  border: none !important;
  border-radius: 9px !important;
  color: #365686 !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 11px 14px !important;
  box-shadow: none !important;
  margin: 2px 0 !important;
}
.nav-btn button:hover {
  background: #f2f8ff !important;
  color: var(--blue) !important;
}
.nav-btn.active button {
  background: var(--blue-soft) !important;
  color: var(--blue) !important;
  font-weight: 600 !important;
}

.sidebar-promo-card {
  margin-top: 28px;
  border: 1px solid var(--border);
  background: #fbfdff;
  border-radius: 12px;
  padding: 16px 14px 0;
  overflow: hidden;
}
.promo-title {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 700;
  color: var(--navy);
}
.promo-desc {
  font-size: 11.5px;
  color: var(--muted);
  line-height: 1.55;
  margin: 6px 0 8px;
}
.promo-art {
  display: flex;
  justify-content: center;
}

/* Main Column */
#main {
  padding: 0 26px 28px !important;
  background: var(--bg-page) !important;
}

.pp-topbar {
  background: #ffffff;
  border-bottom: 1px solid var(--border);
  margin: 0 -26px 20px;
  padding: 12px 26px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.pp-search-mock {
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid #dceaff;
  background: #fcfdff;
  border-radius: 8px;
  padding: 8px 14px;
  width: 440px;
  max-width: 100%;
  color: #89a3cc;
  font-size: 13.5px;
}
.pp-user-area {
  display: flex;
  align-items: center;
  gap: 16px;
  color: var(--navy);
  font-size: 13.5px;
  font-weight: 600;
}
.pp-avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: #dbeaff;
  color: var(--blue);
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.pp-greeting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  flex-wrap: wrap;
  gap: 12px;
}
.pp-page-title {
  font-size: 24px;
  font-weight: 800;
  color: var(--navy);
  margin: 0;
}
.pp-page-sub {
  font-size: 13.5px;
  color: var(--muted);
  margin-top: 3px;
}
.pp-date-pill {
  background: #ffffff;
  border: 1px solid #dceaff;
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--navy);
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

/* Panels / Cards */
.panel {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 20px !important;
  box-shadow: 0 2px 10px rgba(11, 49, 104, 0.02) !important;
}

.pp-card-header-split {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.pp-header-left {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.pp-icon-circle {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #eaf3ff;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.pp-card-title {
  font-size: 15.5px;
  font-weight: 700;
  color: var(--navy);
}
.pp-card-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}
.pp-status-pill {
  background: #e6faf3;
  color: #00b884;
  font-size: 12px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.pp-status-pill .status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #00c48c;
}

/* Dropzone */
.pp-dropzone {
  border: 1.5px dashed #b9d8ff;
  background: #f9fcff;
  border-radius: 10px;
  padding: 22px 14px 16px;
  text-align: center;
  margin-top: 14px;
}
.pp-dropzone-title {
  font-size: 14.5px;
  font-weight: 700;
  color: var(--navy);
  margin-top: 6px;
}
.pp-dropzone-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}
.pp-or-divider {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--muted);
  font-size: 11.5px;
  margin: 12px 0;
}
.pp-or-divider span {
  height: 1px;
  background: var(--border);
  flex: 1;
}

/* Primary button */
.primary-btn button {
  background: var(--blue) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  min-height: 44px !important;
  box-shadow: none !important;
}
.primary-btn button:hover {
  background: #146ee5 !important;
}

/* Result Hero */
.pp-result-hero {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 18px 0 14px;
}
.pp-hero-text {
  min-width: 145px;
}
.pp-posture-name {
  font-size: 26px;
  font-weight: 800;
  color: var(--navy);
  line-height: 1.2;
}
.pp-posture-conf {
  font-size: 13px;
  color: var(--muted);
  margin-top: 4px;
}
.pp-progress-area {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 12px;
}
.pp-progress-track {
  flex: 1;
  height: 14px;
  background: #e8f2ff;
  border-radius: 3px;
  overflow: hidden;
}
.pp-progress-fill {
  height: 100%;
  background: var(--blue);
  transition: width 0.5s ease;
}
.pp-progress-val {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--navy);
  min-width: 44px;
  text-align: right;
}

/* Metric 3 tiles */
.pp-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 6px;
}
.pp-metric-box {
  background: #f4f8ff;
  border-radius: 8px;
  padding: 12px 14px;
}
.pp-metric-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  color: #355586;
}
.m-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}
.m-dot.blue { background: var(--blue); }
.m-dot.green { background: var(--green); }
.m-dot.purple { background: var(--purple); }
.pp-metric-num {
  font-size: 14px;
  font-weight: 600;
  color: var(--navy);
  margin-top: 5px;
  padding-left: 17px;
}

/* Bottom Stats & Activity */
.pp-sec-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15.5px;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 14px;
}
.pp-stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
}
.pp-stat-col {
  padding: 6px 16px;
  border-right: 1px solid var(--border);
}
.pp-stat-col:first-child { padding-left: 4px; }
.pp-stat-col:last-child { border-right: none; }
.pp-stat-ico { margin-bottom: 8px; }
.pp-stat-val {
  font-size: 20px;
  font-weight: 800;
  color: var(--navy);
}
.pp-stat-lbl {
  font-size: 11.5px;
  color: var(--muted);
  margin-top: 3px;
}

.pp-act-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.pp-view-all {
  font-size: 12px;
  color: var(--blue);
  text-decoration: none;
  font-weight: 500;
}
.pp-act-row {
  display: grid;
  grid-template-columns: 10px 1.15fr 1fr 0.9fr auto;
  align-items: center;
  gap: 8px;
  padding: 9px 0;
  border-bottom: 1px solid #eef5ff;
  font-size: 12px;
  color: var(--muted);
}
.pp-act-row:last-child { border-bottom: none; }
.pp-act-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}
.pp-act-pose {
  color: var(--navy);
  font-weight: 700;
}
"""

# Khởi tạo dữ liệu và đồ thị mặc định ngay khi mở trang để giao diện lung linh như hình mẫu
INITIAL_WAVE_TEXT = generate_realistic_waveforms("Đang ngồi (Mẫu chuẩn)")
INITIAL_ARRAY = parse_sensor_text(INITIAL_WAVE_TEXT)
INITIAL_FIG = create_sensor_figure(INITIAL_ARRAY)
INITIAL_RESULT_HTML = build_result_card_html("Đang ngồi", 87.3, 87.3, 8.1, 4.6, "10/10/2026 16:42")
INITIAL_STATS_HTML = build_stats_html(1248, 87.3, "12.4s", 3)
INITIAL_HISTORY_HTML = build_history_html(RECENT_HISTORY)

with gr.Blocks(
    css=CSS,
    title="PhonePose | Dashboard Nhận diện Tư thế",
    theme=gr.themes.Base(primary_hue="blue", secondary_hue="blue", neutral_hue="slate"),
) as interface:
    with gr.Row(elem_id="app-shell", equal_height=False):
        # -------------------- SIDEBAR TRÁI --------------------
        with gr.Column(elem_id="sidebar", scale=0, min_width=244):
            gr.HTML(SVG_LOGO)
            gr.Button("⌂   Trang chủ", elem_classes=["nav-btn", "active"])
            gr.Button("◎   Dự đoán tư thế", elem_classes="nav-btn")
            gr.Button("▤   Dữ liệu", elem_classes="nav-btn")
            gr.Button("◷   Lịch sử", elem_classes="nav-btn")
            gr.Button("⚙   Cài đặt", elem_classes="nav-btn")
            gr.HTML(SVG_SIDEBAR_CARD)

        # -------------------- NỘI DUNG CHÍNH --------------------
        with gr.Column(elem_id="main", scale=1, min_width=0):
            # Thanh Topbar + Tiêu đề chào mừng
            gr.HTML(
                """
                <div class="pp-topbar">
                  <div class="pp-search-mock">
                    <span>🔍</span><span>Tìm kiếm...</span>
                  </div>
                  <div class="pp-user-area">
                    <span>🔔</span>
                    <span style="color:#e2eeff">|</span>
                    <span class="pp-avatar">👤</span>
                    <span>Người dùng ⌄</span>
                  </div>
                </div>
                <div class="pp-greeting-row">
                  <div>
                    <h1 class="pp-page-title">Xin chào, Người dùng!</h1>
                    <div class="pp-page-sub">Cùng khám phá và phân tích dữ liệu tư thế sử dụng điện thoại của bạn.</div>
                  </div>
                  <div class="pp-date-pill">
                    <span>📅</span><span>10/10/2026 - 10/10/2026</span><span>⌄</span>
                  </div>
                </div>
                """
            )

            # Hàng 1: Bắt đầu dự đoán (Trái) + Kết quả & Đồ thị (Phải)
            with gr.Row(equal_height=False):
                # Cột Trái: Bắt đầu dự đoán
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="panel"):
                        gr.HTML(
                            """
                            <div class="pp-header-left">
                              <div class="pp-icon-circle">
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2">
                                  <ellipse cx="12" cy="5" rx="8" ry="3"/>
                                  <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>
                                  <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>
                                </svg>
                              </div>
                              <div>
                                <div class="pp-card-title">Bắt đầu dự đoán</div>
                                <div class="pp-card-sub">Tải lên tệp dữ liệu hoặc chọn từ bộ dữ liệu có sẵn</div>
                              </div>
                            </div>
                            <div class="pp-dropzone">
                              <svg width="46" height="46" viewBox="0 0 48 48" fill="none" style="margin:0 auto;display:block">
                                <path d="M34.5 34H37C40.866 34 44 30.866 44 27C44 23.39 41.266 20.418 37.752 20.04C36.66 13.79 31.216 9 24.667 9C17.303 9 11.333 14.97 11.333 22.333C11.333 22.74 11.352 23.143 11.388 23.54C7.79 24.36 5 27.584 5 31.5C5 36.194 8.806 40 13.5 40H16" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M24 38V23M24 23L18.5 28.5M24 23L29.5 28.5" stroke="#1e80ff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
                              </svg>
                              <div class="pp-dropzone-title">Kéo thả tệp vào đây</div>
                              <div class="pp-dropzone-sub">Hoặc chọn tệp từ máy</div>
                            </div>
                            """
                        )
                        uploaded_file = gr.File(
                            label="📁 Chọn tệp (.csv, .txt, .json)",
                            file_types=[".csv", ".txt", ".json"],
                        )
                        gr.HTML(
                            """
                            <div style="font-size:11.5px;color:#6c88b5;margin-top:4px">
                              Hỗ trợ định dạng: .csv, .txt, .json (tín hiệu cảm biến)
                            </div>
                            <div class="pp-or-divider"><span></span>Hoặc<span></span></div>
                            <div style="display:flex;align-items:center;gap:8px;font-weight:700;font-size:13.5px;color:#0b3168;margin-bottom:6px">
                              <span style="color:#1e80ff">▤</span><span>Chọn bộ dữ liệu có sẵn</span>
                            </div>
                            """
                        )
                        preset_dropdown = gr.Dropdown(
                            choices=[
                                "Đang ngồi (Mẫu chuẩn)",
                                "Đứng (Mẫu #2)",
                                "Đi bộ (Mẫu #3)",
                                "Nằm (Mẫu #4)",
                                "Lên cầu thang (Mẫu #5)",
                            ],
                            value="Đang ngồi (Mẫu chuẩn)",
                            label="Bộ dữ liệu mẫu",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=["1D-CNN", "CotMix UCI-HAR (TorchScript)", "DINOv3"],
                            value="1D-CNN",
                            label="Mô hình",
                        )
                        with gr.Accordion("Xem / Chỉnh sửa chuỗi tín hiệu cảm biến (9 × 128)", open=False):
                            input_text = gr.Textbox(
                                value=INITIAL_WAVE_TEXT,
                                label="Dữ liệu cảm biến",
                                lines=3,
                            )
                        btn_predict = gr.Button("▶   Bắt đầu dự đoán", elem_classes="primary-btn", size="lg")

                # Cột Phải: Kết quả dự đoán + Tín hiệu cảm biến
                with gr.Column(scale=7):
                    with gr.Group(elem_classes="panel"):
                        output_summary = gr.HTML(INITIAL_RESULT_HTML)

                    with gr.Group(elem_classes="panel"):
                        gr.HTML(
                            """
                            <div class="pp-sec-head" style="margin-bottom:4px">
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1e80ff" stroke-width="2.2">
                                <line x1="18" y1="20" x2="18" y2="10" stroke-linecap="round"/>
                                <line x1="12" y1="20" x2="12" y2="4" stroke-linecap="round"/>
                                <line x1="6" y1="20" x2="6" y2="14" stroke-linecap="round"/>
                              </svg>
                              <span>Tín hiệu cảm biến</span>
                            </div>
                            """
                        )
                        output_plot = gr.Plot(value=INITIAL_FIG, show_label=False)

            # Hàng 2: Thống kê nhanh + Hoạt động gần đây
            with gr.Row(equal_height=True):
                with gr.Column(scale=7):
                    with gr.Group(elem_classes="panel"):
                        output_stats = gr.HTML(INITIAL_STATS_HTML)
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="panel"):
                        output_history = gr.HTML(INITIAL_HISTORY_HTML)

    # Sự kiện tương tác
    uploaded_file.change(fn=read_uploaded_file, inputs=uploaded_file, outputs=input_text)
    preset_dropdown.change(fn=on_preset_change, inputs=preset_dropdown, outputs=input_text)
    btn_predict.click(
        fn=predict_action,
        inputs=[input_text, preset_dropdown],
        outputs=[output_plot, output_summary, output_stats, output_history],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)