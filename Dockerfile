# ==========================================
# Stage 1: Build Dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compiler toolchain needed for C-extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to isolated user folder
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Hardened Production Runtime
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Create dedicated non-root service group and user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Copy only the compiled Python packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and golden baseline
COPY --chown=appuser:appgroup ./app /app/app
COPY --chown=appuser:appgroup ./baseline /app/baseline

# Configure environment and drop root privileges
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

USER 10001:10001

EXPOSE 8000

# Health check matching Kubernetes readiness probe pattern
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/readiness')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]