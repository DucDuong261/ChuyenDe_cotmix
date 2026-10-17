FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Cài đặt PyTorch bản CPU siêu nhẹ trước để chống tràn RAM
RUN pip install --upgrade pip && pip install torch --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]