FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt setup.py ./
COPY ct_dgnn ./ct_dgnn
COPY scripts ./scripts
COPY configs ./configs

RUN pip install -r requirements.txt \
 && pip install -e .

EXPOSE 8000
CMD ["python", "-m", "deployment.robustidps_integration"]
