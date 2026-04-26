FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
COPY starter.py .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir pandas pyarrow scikit-learn==1.5.0 && \
    python -c "import sklearn; print(sklearn.__version__)"

ENTRYPOINT ["python", "starter.py"]