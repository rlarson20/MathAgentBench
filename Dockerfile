FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency management
RUN "pip install --no-cache-dir uv"

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN "uv pip install --system -e ."

# Copy source code
COPY src/ src/
COPY benchmarks/ benchmarks/

# Create results directory
RUN "mkdir -p results"

ENTRYPOINT ["mathagent"]
