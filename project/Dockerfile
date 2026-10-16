FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
COPY . .
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
CMD ["bash", "scripts/run_smoke.sh"]
