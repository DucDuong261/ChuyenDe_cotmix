import gradio as gr
import torch
import numpy as np
import sys
import os

# Thêm thư mục hiện tại vào sys.path để có thể load class cấu trúc mô hình
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Tải mô hình
device = torch.device("cpu")
try:
    # Load file mô hình .pkl
    model = torch.load("cotmix_uci_har.pkl", map_location=device)
    model.eval()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# 2. Hàm dự đoán
def predict_action(input_data):
    if not model_loaded:
        return f"Lỗi tải mô hình: {error_msg}.\n(Đảm bảo bạn đã tải lên các file thuật toán như models/, algorithms.py)"
    
    try:
        # Chuyển chuỗi text đầu vào thành mảng số
        data_list = [float(x.strip()) for x in input_data.split(',')]
        
        # Định dạng shape đầu vào UCI-HAR (1 batch, 9 channels, 128 timesteps)
        expected_length = 9 * 128
        if len(data_list) < expected_length:
            data_list.extend([0.0] * (expected_length - len(data_list))) # Padding nếu thiếu
        elif len(data_list) > expected_length:
            data_list = data_list[:expected_length] # Cắt bớt nếu thừa
            
        data_array = np.array(data_list).reshape(1, 9, 128)
        tensor_data = torch.tensor(data_array, dtype=torch.float32).to(device)
        
        # Chạy qua mô hình
        with torch.no_grad():
            features = model.feature_extractor(tensor_data)
            predictions = model.classifier(features)
            predicted_class = predictions.argmax(dim=1).item()
            
        # Mapping nhãn hành động tương ứng của bộ UCI-HAR
        labels = {
            0: "Đi bộ (Walking)", 
            1: "Đi bộ lên cầu thang (Walking Upstairs)", 
            2: "Đi bộ xuống cầu thang (Walking Downstairs)", 
            3: "Ngồi (Sitting)", 
            4: "Đứng (Standing)", 
            5: "Nằm (Laying)"
        }
        action = labels.get(predicted_class, f"Nhãn ID {predicted_class}")
            
        return f"🤖 Kết quả dự đoán: {action}"
    except Exception as e:
        return f"Lỗi xử lý dữ liệu: {str(e)}"

# Hàm tạo dữ liệu ngẫu nhiên để test nhanh
def generate_dummy_data():
    dummy = np.random.randn(9 * 128).round(4)
    return ",".join(map(str, dummy))

# 3. Tạo giao diện UI bằng Gradio
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏃 Nhận dạng hành động bằng cảm biến (UCI-HAR)")
    gr.Markdown("Demo mô hình 1D CNN dự đoán hành động từ dữ liệu gia tốc kế & con quay hồi chuyển.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=8, 
                label="Dữ liệu cảm biến (1152 con số, cách nhau bằng dấu phẩy)",
                placeholder="Nhấn nút bên dưới để tự động điền dữ liệu test..."
            )
            with gr.Row():
                btn_dummy = gr.Button("🎲 Sinh dữ liệu cảm biến ngẫu nhiên")
                btn_predict = gr.Button("🚀 Dự đoán Hành động", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Kết quả trả về", lines=4)
            
    btn_dummy.click(fn=generate_dummy_data, inputs=[], outputs=[input_text])
    btn_predict.click(fn=predict_action, inputs=[input_text], outputs=[output_text])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)