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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cotmix_uci_har.pkl")
device = torch.device("cpu")
model = None
model_error = None

LABELS = {
    0: "Đi bộ (Walking)",
    1: "Lên cầu thang (Upstairs)",
    2: "Xuống cầu thang (Downstairs)",
    3: "Ngồi (Sitting)",
    4: "Đứng (Standing)",
    5: "Nằm (Laying)",
}

LABEL_SHORT = {
    "Đi bộ (Walking)": "Đi bộ",
    "Lên cầu thang (Upstairs)": "Lên cầu thang",
    "Xuống cầu thang (Downstairs)": "Xuống cầu thang",
    "Ngồi (Sitting)": "Ngồi",
    "Đứng (Standing)": "Đứng",
    "Nằm (Laying)": "Nằm",
}

def load_model_if_needed():
    global model, model_error
    if model is None and model_error is None:
        try:
            # Sửa tên file thành cotmix_uci_har.pt
            model = torch.jit.load("cotmix_uci_har.pt", map_location=device)
            model.eval()
        except Exception as e:
            model_error = str(e)
    return model, model_error


def parse_sensor_text(input_data):
    """Đọc chuỗi số phân tách bằng dấu phẩy, khoảng trắng hoặc xuống dòng."""
    if not input_data or not input_data.strip():
        raise ValueError("Vui lòng nhập dữ liệu cảm biến hoặc tải tệp lên.")
    tokens = input_data.replace("\n", ",").replace("\t", ",").replace(";", ",").split(",")
    values = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Hỗ trợ chuỗi có nhiều số cách nhau bằng dấu cách.
        values.extend(float(part) for part in token.split())
    if not values:
        raise ValueError("Không tìm thấy giá trị số trong dữ liệu.")
    expected = 9 * 128
    if len(values) < expected:
        values.extend([0.0] * (expected - len(values)))
    return np.asarray(values[:expected], dtype=np.float32).reshape(1, 9, 128)


def predict_action(input_data):
    current_model, err = load_model_if_needed()
    if err:
        raise gr.Error(f"Không thể tải mô hình: {err}")
    if current_model is None:
        raise gr.Error("Mô hình chưa sẵn sàng.")

    try:
        data_array = parse_sensor_text(input_data)
        tensor_data = torch.tensor(data_array, dtype=torch.float32, device=device)

        start = time.perf_counter()
        with torch.no_grad():
            features = current_model.feature_extractor(tensor_data)
            logits = current_model.classifier(features)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        elapsed = time.perf_counter() - start

        confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        best_idx = int(np.argmax(probs))
        best_label = LABELS[best_idx]
        confidence = float(probs[best_idx]) * 100

        # Đồ thị cảm biến X/Y/Z theo phong cách giao diện tham chiếu.
        fig, ax = plt.subplots(figsize=(8.4, 2.35), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = np.arange(128)
        ax.plot(x, data_array[0, 0], color="#2583ff", lw=1.25, label="X")
        ax.plot(x, data_array[0, 1], color="#00c99a", lw=1.25, label="Y")
        ax.plot(x, data_array[0, 2], color="#9347ff", lw=1.25, label="Z")
        ax.grid(True, linestyle="-", linewidth=0.45, color="#e4efff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d9e9ff")
        ax.spines["bottom"].set_color("#d9e9ff")
        ax.tick_params(colors="#6483b5", labelsize=7, length=0)
        ax.set_xlabel("Mẫu tín hiệu", fontsize=8, color="#6483b5", labelpad=5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
                  frameon=False, fontsize=8, labelcolor="#6483b5")
        fig.tight_layout(pad=1.2)

        # Dữ liệu hiển thị được cập nhật sau mỗi lần dự đoán.
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        summary = (
            f"<div class='result-summary'>"
            f"<div class='result-illustration'>♙</div>"
            f"<div class='result-main'><div class='result-label'>{LABEL_SHORT.get(best_label, best_label)}</div>"
            f"<div class='muted'>Độ tin cậy: {confidence:.1f}%</div></div>"
            f"<div class='confidence-wrap'><div class='confidence-track'><div class='confidence-fill' style='width:{confidence:.1f}%'></div></div>"
            f"<b>{confidence:.1f}%</b></div></div>"
            f"<div class='prediction-metrics'>"
            f"<div class='metric-tile'><span class='dot blue'></span><span>Đi bộ</span><b>{probs[0]*100:.1f}%</b></div>"
            f"<div class='metric-tile'><span class='dot green'></span><span>Ngồi / Đứng</span><b>{max(probs[3], probs[4])*100:.1f}%</b></div>"
            f"<div class='metric-tile'><span class='dot purple'></span><span>Nằm</span><b>{probs[5]*100:.1f}%</b></div>"
            f"</div><div class='result-foot'>Hoàn thành · {now} · Thời gian xử lý: {elapsed:.3f}s</div>"
        )
        return confidences, fig, summary
    except Exception as exc:
        raise gr.Error(f"Lỗi xử lý dữ liệu: {exc}")


def generate_dummy_data():
    # Dữ liệu giả lập để thử giao diện; không phải mẫu đo thực tế.
    dummy = np.random.default_rng().normal(0, 1, 9 * 128).round(4)
    return ", ".join(map(str, dummy))


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
        # Loại bỏ dòng tiêu đề CSV nếu có; chỉ giữ các giá trị có thể chuyển thành số.
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


# -------------------- Giao diện PhonePose --------------------
CSS = r"""
:root {
  --navy: #0b3478;
  --blue: #2583ff;
  --blue-light: #eaf4ff;
  --line: #d7e8ff;
  --muted: #6383b7;
  --green: #00c99a;
  --purple: #9347ff;
}
body, .gradio-container {
  background: #ffffff !important;
  color: var(--navy) !important;
  font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }
#app-shell { gap: 0 !important; min-height: 100vh; }
#sidebar {
  background: #fbfdff;
  border-right: 1px solid #e1edff;
  padding: 22px 16px 18px !important;
  min-width: 250px !important;
  max-width: 250px !important;
  min-height: 100vh;
}
.logo-mark { color: var(--blue); font-size: 27px; font-weight: 800; letter-spacing: -1px; padding: 2px 12px 24px; }
.logo-mark span { color: var(--navy); }
.nav-btn button {
  width: 100%; text-align: left !important; justify-content: flex-start !important;
  border: 0 !important; background: transparent !important; color: var(--navy) !important;
  box-shadow: none !important; border-radius: 0 !important; padding: 13px 14px !important;
  font-size: 15px !important; margin: 2px 0 !important;
}
.nav-btn.active button { background: #e8f3ff !important; color: var(--blue) !important; font-weight: 700 !important; }
.sidebar-info { border: 1px solid var(--line); padding: 15px; margin-top: 24px; color: var(--muted); font-size: 12px; line-height: 1.7; }
#main { padding: 0 28px 24px !important; min-width: 0; }
#topbar { border-bottom: 1px solid #e4efff; padding: 18px 0 16px; align-items: center; }
.searchbox input { border: 1px solid var(--line) !important; border-radius: 0 !important; box-shadow: none !important; }
.user-label { text-align: right; color: var(--navy); padding: 12px 0; }
.page-title { font-size: 27px; font-weight: 750; color: var(--navy); margin: 20px 0 2px; }
.page-subtitle { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
.panel {
  border: 1px solid var(--line) !important;
  background: #fff !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 18px !important;
}
.panel-title { color: var(--navy); font-size: 16px; font-weight: 750; margin: 0 0 4px; }
.panel-subtitle { color: var(--muted); font-size: 12px; margin-bottom: 15px; }
.upload-drop {
  border: 1px dashed #9dc8ff; padding: 28px 10px 20px; text-align: center;
  min-height: 160px; color: var(--blue); margin: 12px 0;
}
.upload-drop .upload-icon { font-size: 40px; line-height: 1.3; }
.upload-drop strong { display: block; color: var(--navy); font-size: 14px; margin: 4px 0; }
.upload-drop small { color: var(--muted); }
.primary-btn button {
  border-radius: 0 !important; background: var(--blue) !important; color: white !important;
  border: 1px solid var(--blue) !important; box-shadow: none !important; font-weight: 700 !important;
  min-height: 44px;
}
.secondary-btn button {
  border-radius: 0 !important; background: white !important; color: var(--blue) !important;
  border: 1px solid var(--blue) !important; box-shadow: none !important;
}
select, input, textarea, .wrap, .form, .block {
  box-shadow: none !important;
}
input, textarea, select, .gradio-dropdown, .gradio-textbox {
  border-radius: 0 !important;
}
.gradio-dropdown, .gradio-textbox, .gradio-file, .gradio-plot {
  border-color: var(--line) !important;
}
.result-summary { display:flex; align-items:center; gap:18px; padding:20px 4px 18px; }
.result-illustration { color:var(--blue); font-size:42px; width:72px; text-align:center; }
.result-main { min-width:130px; }
.result-label { font-size:24px; font-weight:800; color:var(--navy); margin-bottom:6px; }
.muted { color:var(--muted); font-size:13px; }
.confidence-wrap { display:flex; align-items:center; gap:12px; flex:1; min-width:100px; color:var(--navy); font-size:12px; }
.confidence-track { background:#e7f2ff; height:14px; flex:1; }
.confidence-fill { height:14px; background:var(--blue); }
.prediction-metrics { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:8px; }
.metric-tile { background:#f3f8ff; padding:14px 12px; display:grid; grid-template-columns:12px 1fr; gap:6px 8px; color:var(--navy); font-size:12px; }
.metric-tile b { grid-column:2; font-size:13px; font-weight:500; }
.dot { width:10px; height:10px; border-radius:50%; margin-top:3px; }
.dot.blue { background:var(--blue); } .dot.green { background:var(--green); } .dot.purple { background:var(--purple); }
.result-foot { color:var(--muted); font-size:11px; margin-top:14px; border-top:1px solid #edf4ff; padding-top:10px; }
.stat-row { display:grid; grid-template-columns:repeat(4,1fr); }
.stat-item { padding:8px 14px; border-right:1px solid var(--line); }
.stat-item:last-child { border-right:0; }
.stat-icon { color:var(--blue); font-size:20px; }
.stat-value { color:var(--navy); font-size:17px; font-weight:750; margin:5px 0; }
.stat-caption { color:var(--muted); font-size:11px; }
.activity-line { display:grid; grid-template-columns:10px 1fr 1fr 1fr auto; gap:8px; padding:9px 0; border-bottom:1px solid #edf4ff; color:var(--muted); font-size:10px; }
.activity-line:last-child { border-bottom:0; }
.activity-dot { width:7px; height:7px; background:var(--blue); border-radius:50%; margin-top:3px; }
.section-label { color:var(--navy); font-weight:700; font-size:14px; margin:10px 0; }
#results-col, #left-col { gap:16px !important; }
.gradio-plot { border:0 !important; }
@media (max-width: 1000px) {
  #sidebar { min-width:190px !important; max-width:190px !important; }
  #main { padding:0 16px 20px !important; }
  .result-summary { flex-wrap:wrap; }
  .confidence-wrap { flex-basis:100%; }
}
@media (max-width: 760px) {
  #app-shell { flex-direction:column !important; }
  #sidebar { min-width:100% !important; max-width:100% !important; min-height:auto; border-right:0; border-bottom:1px solid var(--line); }
  #main { padding:0 12px 18px !important; }
  .prediction-metrics, .stat-row { grid-template-columns:1fr 1fr; }
}
"""

with gr.Blocks(css=CSS, title="PhonePose | Dashboard", theme=gr.themes.Base(
    primary_hue="blue", secondary_hue="blue", neutral_hue="slate"
)) as interface:
    with gr.Row(elem_id="app-shell", equal_height=False):
        # Thanh điều hướng bên trái
        with gr.Column(elem_id="sidebar", scale=0, min_width=250):
            gr.HTML("<div class='logo-mark'>♧ <span>Phone</span>Pose</div>")
            gr.Button("⌂   Trang chủ", elem_classes=["nav-btn", "active"])
            gr.Button("◎   Dự đoán tư thế", elem_classes="nav-btn")
            gr.Button("▤   Dữ liệu", elem_classes="nav-btn")
            gr.Button("◷   Lịch sử", elem_classes="nav-btn")
            gr.Button("⚙   Cài đặt", elem_classes="nav-btn")
            gr.HTML(
                "<div class='sidebar-info'><b style='color:#0b3478;font-size:14px'>▯ &nbsp; PhonePose</b>"
                "<br>Nhận diện tư thế sử dụng điện thoại từ dữ liệu cảm biến"
                "<div style='text-align:center;color:#2583ff;font-size:64px;line-height:1.5;margin-top:16px'>♙</div></div>"
            )

        with gr.Column(elem_id="main", scale=1, min_width=0):
            with gr.Row(elem_id="topbar"):
                search = gr.Textbox(placeholder="⌕   Tìm kiếm...", show_label=False, scale=8,
                                    elem_classes="searchbox", container=False)
                gr.HTML("<div style='font-size:22px;text-align:center;padding:8px 14px;color:#0b3478'>♧</div>")
                gr.HTML("<div class='user-label'>● &nbsp; Người dùng &nbsp;⌄</div>")

            gr.HTML("<div class='page-title'>Xin chào, Người dùng!</div>"
                    "<div class='page-subtitle'>Cùng khám phá và phân tích dữ liệu tư thế sử dụng điện thoại của bạn.</div>")

            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_id="left-col"):
                    with gr.Group(elem_classes="panel"):
                        gr.HTML("<div class='panel-title'>◉ &nbsp; Bắt đầu dự đoán</div>"
                                "<div class='panel-subtitle'>Tải lên tệp dữ liệu hoặc chọn từ bộ dữ liệu có sẵn</div>")
                        gr.HTML("<div class='upload-drop'><div class='upload-icon'>⇧</div>"
                                "<strong>Kéo thả tệp vào đây</strong><small>Hoặc chọn tệp từ máy</small></div>")
                        uploaded_file = gr.File(label="Tải tệp dữ liệu (.csv, .txt, .json)", file_types=[".csv", ".txt", ".json"])
                        gr.Markdown("<div style='font-size:11px;color:#6383b7'>Hỗ trợ định dạng: .csv, .txt, .json (tín hiệu cảm biến)</div>")
                        gr.HTML("<div style='display:flex;align-items:center;gap:12px;color:#6383b7;font-size:12px;margin:12px 0'><span style='height:1px;background:#d7e8ff;flex:1'></span>Hoặc<span style='height:1px;background:#d7e8ff;flex:1'></span></div>")
                        gr.HTML("<div class='section-label'>▤ &nbsp; Chọn bộ dữ liệu có sẵn</div>")
                        input_text = gr.Textbox(
                            label="Dữ liệu cảm biến (9 × 128 giá trị)",
                            placeholder="Dữ liệu từ tệp sẽ xuất hiện tại đây; hoặc nhập chuỗi số, phân cách bằng dấu phẩy.",
                            lines=4,
                                        )
                        with gr.Row():
                            btn_dummy = gr.Button("Tạo dữ liệu mẫu", elem_classes="secondary-btn", scale=1)
                            model_dropdown = gr.Dropdown(
                                choices=["1D-CNN (Mặc định)", "DINOv3 (chỉ hiển thị lựa chọn)"],
                                value="1D-CNN (Mặc định)", label="Mô hình", scale=2
                            )
                        btn_predict = gr.Button("▶   Bắt đầu dự đoán", elem_classes="primary-btn", size="lg")
                        gr.Markdown("<div style='font-size:11px;color:#6383b7'>Dữ liệu đầu vào được chuẩn hóa về 9 kênh × 128 bước thời gian.</div>")

                with gr.Column(scale=6, elem_id="results-col"):
                    with gr.Group(elem_classes="panel"):
                        gr.HTML("<div class='panel-title'>▯ &nbsp; Kết quả dự đoán</div>"
                                "<div class='panel-subtitle'>Kết quả mới nhất từ mô hình</div>")
                        output_summary = gr.HTML(
                            "<div class='result-summary'><div class='result-illustration'>♙</div>"
                            "<div class='result-main'><div class='result-label'>Chưa có kết quả</div>"
                            "<div class='muted'>Tải dữ liệu lên và bấm “Bắt đầu dự đoán”.</div></div></div>"
                        )
                        output_label = gr.Label(num_top_classes=6, label="Xác suất theo từng hoạt động")
                    with gr.Group(elem_classes="panel"):
                        gr.HTML("<div class='panel-title'>▥ &nbsp; Tín hiệu cảm biến</div>"
                                "<div class='panel-subtitle'>Biểu diễn ba trục gia tốc X, Y, Z</div>")
                        output_plot = gr.Plot(label="Tín hiệu X, Y, Z", show_label=False)

            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="panel"):
                        gr.HTML("<div class='panel-title'>▥ &nbsp; Thống kê nhanh</div>"
                                "<div class='stat-row'>"
                                "<div class='stat-item'><div class='stat-icon'>▯</div><div class='stat-value'>1,248</div><div class='stat-caption'>Tổng số mẫu</div></div>"
                                "<div class='stat-item'><div class='stat-icon'>◔</div><div class='stat-value' id='accuracy'>87.3%</div><div class='stat-caption'>Tỷ lệ dự đoán đúng*</div></div>"
                                "<div class='stat-item'><div class='stat-icon'>◷</div><div class='stat-value' id='runtime'>0.12s</div><div class='stat-caption'>Thời gian xử lý tham khảo</div></div>"
                                "<div class='stat-item'><div class='stat-icon'>▤</div><div class='stat-value'>6</div><div class='stat-caption'>Lớp hoạt động</div></div>"
                                "</div><div style='color:#8aa0c2;font-size:10px;margin-top:8px'>*Số liệu minh họa giao diện, không phải kết quả đánh giá mô hình hiện tại.</div>")
                with gr.Column(scale=4):
                    with gr.Group(elem_classes="panel"):
                        gr.HTML("<div style='display:flex;justify-content:space-between;align-items:center'>"
                                "<div class='panel-title'>◷ &nbsp; Hoạt động gần đây</div>"
                                "<span style='font-size:11px;color:#2583ff'>Xem tất cả →</span></div>"
                                "<div class='activity-line'><span class='activity-dot'></span><span>Chưa có dữ liệu</span><span>—</span><span>Chờ dự đoán</span><span>—</span></div>"
                                "<div class='activity-line'><span class='activity-dot' style='background:#00c99a'></span><span>Chạy mô hình</span><span>—</span><span>Chưa thực hiện</span><span>—</span></div>"
                                "<div class='activity-line'><span class='activity-dot' style='background:#9347ff'></span><span>Phân tích tín hiệu</span><span>—</span><span>Chưa thực hiện</span><span>—</span></div>")

    uploaded_file.change(fn=read_uploaded_file, inputs=uploaded_file, outputs=input_text)
    btn_dummy.click(fn=generate_dummy_data, inputs=[], outputs=input_text)
    btn_predict.click(
        fn=predict_action,
        inputs=input_text,
        outputs=[output_label, output_plot, output_summary]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
