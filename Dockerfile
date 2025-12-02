FROM huggingface/lerobot-gpu:latest

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --upgrade pip && \
    uv pip install .[modal]
