FROM python:3.11-slim

WORKDIR /app

# System deps: git for cloning repos, grep for search
RUN apt-get update && apt-get install -y \
    git \
    grep \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY openenv.yaml .
COPY rlm_forge/ rlm_forge/

# Install Python deps
RUN pip install --no-cache-dir -e .

# Pre-install common test dependencies for target repos
RUN pip install --no-cache-dir pytest text-unidecode freezegun

# AMENDMENT 2: Pre-clone curated repos to avoid network I/O on every reset()
RUN mkdir -p /app/repos && \
    git clone --depth=1 https://github.com/un33k/python-slugify /app/repos/python-slugify && \
    git clone --depth=1 https://github.com/python-humanize/humanize /app/repos/humanize

# Install curated repo dependencies
RUN pip install --no-cache-dir -e /app/repos/python-slugify || true
RUN pip install --no-cache-dir -e /app/repos/humanize || true

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV RLM_FORGE_PRE_CLONED_DIR=/app/repos

CMD ["python", "-m", "uvicorn", "rlm_forge.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
