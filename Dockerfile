FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false
WORKDIR /projet_final
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install poetry 
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root
COPY . . 
EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

