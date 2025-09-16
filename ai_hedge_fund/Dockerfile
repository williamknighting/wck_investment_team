# Multi-stage build for AI Hedge Fund System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    libpq-dev \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r hedge_fund && useradd -r -g hedge_fund -m hedge_fund

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e .[dev,ml,backtesting,data,visualization]

# Copy source code
COPY --chown=hedge_fund:hedge_fund . .

# Create necessary directories
RUN mkdir -p logs data_cache && \
    chown -R hedge_fund:hedge_fund logs data_cache

# Switch to non-root user
USER hedge_fund

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "ai_hedge_fund.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install -e .[production]

# Copy source code
COPY --chown=hedge_fund:hedge_fund src/ ./src/
COPY --chown=hedge_fund:hedge_fund config/ ./config/
COPY --chown=hedge_fund:hedge_fund strategies/ ./strategies/

# Create necessary directories
RUN mkdir -p logs data_cache && \
    chown -R hedge_fund:hedge_fund logs data_cache

# Switch to non-root user
USER hedge_fund

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "gunicorn", "ai_hedge_fund.server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Testing stage
FROM development as testing

# Install additional testing tools
RUN pip install pytest-xdist pytest-benchmark

# Copy test files
COPY --chown=hedge_fund:hedge_fund tests/ ./tests/

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=ai_hedge_fund", "--cov-report=html"]

# Jupyter stage for research
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install jupyterlab jupyter-lsp python-lsp-server[all] \
    jupyterlab-git jupyterlab-github

# Copy notebooks
COPY --chown=hedge_fund:hedge_fund notebooks/ ./notebooks/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Deployment stage
FROM production as deployment

# Add deployment-specific configurations
COPY --chown=hedge_fund:hedge_fund docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["server"]