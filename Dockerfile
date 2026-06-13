# ─── Streamlit Dockerfile ────────────────────────────────────────────────────
# Uses a slim Python 3.11 base image to keep the container lightweight.
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (leverages Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (no cache to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application source into the container
COPY . .

# Expose Streamlit's default port so the host can reach it
EXPOSE 8501

# Health-check: verify Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app; server.address=0.0.0.0 makes it reachable from outside
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
