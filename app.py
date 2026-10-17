import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os

# Tăng giới hạn đệ quy để chống lỗi khi load mô hình pkl
sys.setrecursionlimit(50000)
torch.set_num_threads(1)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Tải mô hình
device = torch.device("cpu")
try:
    model = torch.load("cotmix_uci_har.pkl", map_location=device, weights_only=False)
    model.eval()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# Danh sách nhãn
LABELS = {
    0: "Đi bộ (Walking)", 
    1: "Lên cầu thang (Upstairs)", 
    2: "Xuống cầu thang (Downstairs)", 
    3: "Ngồi (Sitting)", 
    4: "Đứng (Standing)", 
    5: "Nằm (Laying)"
}

# 2. Hàm dự đoán và vẽ biểu đồ
def predict_action(input_data):
    if not model_loaded:
        raise gr.Error(f"Lỗi tải mô hình: {error_msg}")
    
    try:
        data_list = [float(x.strip()) for x in input_data.split(',')]
        expected_length = 9 * 128
        if len(data_list) < expected_length:
            data_list.extend([0.0] * (expected_length - len(data_list)))
        elif len(data_list) > expected_length:
            data_list = data_list[:expected_length]
            
        data_array = np.array(data_list).reshape(1, 9, 128)
        tensor_data = torch.tensor(data_array, dtype=torch.float32).to(device)
        
        # Chạy suy luận
        with torch.no_grad():
            features = model.feature_extractor(tensor_data)
            logits = model.classifier(features)
            # Tính phần trăm độ tin cậy
            probs = F.softmax(logits, dim=1).numpy()[0]
            
        # Tạo Dictionary để hiển thị thanh phần trăm (gr.Label)
        confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        
        # Vẽ biểu đồ tín hiệu cảm biến (Lấy 3 kênh đầu tiên: Gia tốc X, Y, Z)
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('white')
        ax.plot(data_array[0, 0, :], label='X', color='#1f77b4', linewidth=1.5)
        ax.plot(data_array[0, 1, :], label='Y', color='#2ca02c', linewidth=1.5)
        ax.plot(data_array[0, 2, :], label='Z', color='#9467bd', linewidth=1.5)
        ax.set_title("Tín hiệu cảm biến (Accel X, Y, Z)", fontsize=10, color='gray')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
            
        return confidences, fig
    except Exception as e:
        raise gr.Error(f"Lỗi xử lý dữ liệu: {str(e)}")

# Hàm tạo dữ liệu giả
def generate_dummy_data():
    dummy = np.random.randn(9 * 128).round(4)
    return ",".join(map(str, dummy))

# 3. Giao diện Gradio mô phỏng PhonePose
custom_css = """
body { background-color: #f4f7f6; }
.gradio-container { max-width: 1200px !important; margin: auto; }
.panel-box { border-radius: 12px; padding: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.header-text { color: #1e3a8a; font-weight: bold; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚶‍♂️ PhonePose Dashboard\n*Cùng khám phá và phân tích dữ liệu tư thế sử dụng điện thoại của bạn.*", elem_classes="header-text")
    
    with gr.Row():
        # Cột Trái: Nhập liệu (Mô phỏng cột "Bắt đầu dự đoán")
        with gr.Column(scale=4, elem_classes="panel-box"):
            gr.Markdown("### 🗄️ Bắt đầu dự đoán\n*Tải lên tệp dữ liệu hoặc chọn từ bộ dữ liệu có sẵn*")
            
            with gr.Tabs():
                with gr.TabItem("Nhập chuỗi văn bản"):
                    input_text = gr.Textbox(lines=6, label="Dữ liệu cảm biến CSV", placeholder="Nhập chuỗi số...")
                    btn_dummy = gr.Button("🎲 Tạo dữ liệu mẫu", size="sm")
                with gr.TabItem("Tải tệp lên (File)"):
                    gr.File(label="Kéo thả tệp vào đây")
                    gr.Markdown("*Lưu ý: Tính năng đọc file đang được phát triển, vui lòng dùng tab văn bản.*")
            
            model_dropdown = gr.Dropdown(choices=["1D-CNN (Mặc định)", "DINOv3"], value="1D-CNN (Mặc định)", label="Mô hình")
            btn_predict = gr.Button("▶ Bắt đầu dự đoán", variant="primary", size="lg")
            
            with gr.Row():
                gr.Markdown("**Thống kê nhanh**\n- Tổng số mẫu: 1,248\n- Thời gian xử lý: ~0.12s")

        # Cột Phải: Kết quả & Biểu đồ
        with gr.Column(scale=7):
            with gr.Group(elem_classes="panel-box"):
                gr.Markdown("### 📱 Kết quả dự đoán")
                # Dùng gr.Label để hiển thị thanh xác suất giống hệt ảnh
                output_label = gr.Label(num_top_classes=3, label="Độ tin cậy (%)")
                
            with gr.Group(elem_classes="panel-box"):
                gr.Markdown("### 📊 Tín hiệu cảm biến")
                output_plot = gr.Plot(label="Đồ thị X, Y, Z")

    # Xử lý sự kiện
    btn_dummy.click(fn=generate_dummy_data, inputs=[], outputs=[input_text])
    btn_predict.click(fn=predict_action, inputs=[input_text], outputs=[output_label, output_plot])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)