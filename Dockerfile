FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Cài PyTorch CPU trước (layer nặng nhất, cache lại nếu không thay đổi)
RUN pip install --user torch --extra-index-url https://download.pytorch.org/whl/cpu

# Chỉ cài thư viện cần cho app.py (không kéo transformers/wandb/sklearn của phần huấn luyện)
COPY --chown=user requirements-app.txt .
RUN pip install --user -r requirements-app.txt

# Chỉ copy những gì app.py dùng: mã nguồn, mô hình TorchScript và dữ liệu mẫu
COPY --chown=user app.py cotmix_uci_har.pt ./
COPY --chown=user CoTMix/samples/ ./samples/

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 7860)}/', timeout=4)"

CMD ["python", "app.py"]
