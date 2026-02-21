FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY scripts/ scripts/
COPY pytorch/ pytorch/
COPY huggingface/ huggingface/
COPY scan-only/ scan-only/
COPY lightning/ lightning/
COPY ray/ ray/
COPY distributed/ distributed/
COPY Makefile .

# Install all workload dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[all]"

# Default: run free-tier validation
CMD ["make", "validate-free"]
