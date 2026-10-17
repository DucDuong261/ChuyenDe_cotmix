# Dùng môi trường Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục và quyền cho user chuẩn của Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Cài đặt thư viện
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ thư mục code hiện tại vào môi trường ảo
COPY --chown=user . .

# Hugging Face yêu cầu mở cổng 7860
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Chạy file app.py
CMD ["python", "app.py"]