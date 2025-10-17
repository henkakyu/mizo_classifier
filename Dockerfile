FROM python:3.12-alpine

# scikit-learn build に必要な依存
RUN apk add --no-cache \
    g++ gcc gfortran musl-dev linux-headers make \
    openblas-dev lapack-dev tini

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/my_model.joblib \
    BLAS=openblas LAPACK=openblas

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir numpy scipy joblib scikit-learn \
&& pip install --no-cache-dir fastapi uvicorn[standard]

COPY app.py ./

EXPOSE 8000
ENTRYPOINT ["/sbin/tini","--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
