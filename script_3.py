# Create deployment configurations for different platforms

# 1. Railway deployment configuration
railway_config = """[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn api_server:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"

[env]
MODEL_CACHE_PATH = "/app/models"
CONFIDENCE_THRESHOLD = "0.5"
MAX_FRAMES_MEMORY = "10"
"""

# Save railway.toml
with open('railway.toml', 'w') as f:
    f.write(railway_config)

# 2. Render deployment configuration (render.yaml)
render_config = """services:
  - type: web
    name: scene-understanding-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn api_server:app --host 0.0.0.0 --port $PORT"
    plan: free
    healthCheckPath: "/health"
    envVars:
      - key: MODEL_CACHE_PATH
        value: "/opt/render/project/models"
      - key: CONFIDENCE_THRESHOLD  
        value: "0.5"
      - key: MAX_FRAMES_MEMORY
        value: "10"
"""

# Save render.yaml
with open('render.yaml', 'w') as f:
    f.write(render_config)

# 3. Dockerfile for containerized deployment (Hugging Face Spaces, Fly.io)
dockerfile_content = '''FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    python3-opencv \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model cache directory
RUN mkdir -p /app/models

# Set environment variables
ENV MODEL_CACHE_PATH=/app/models
ENV CONFIDENCE_THRESHOLD=0.5
ENV MAX_FRAMES_MEMORY=10
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:7860/health || exit 1

# Start command
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
'''

# Save Dockerfile
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

# 4. Hugging Face Spaces configuration
hf_spaces_config = """title: Enhanced Scene Understanding API
emoji: 🎥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
"""

# Save README.md for Hugging Face Spaces
with open('README.md', 'w') as f:
    f.write(hf_spaces_config)

# 5. Fly.io configuration
fly_config = """app = "scene-understanding-api"
kill_signal = "SIGINT"
kill_timeout = 5

[build]
  image = "scene-understanding-api:latest"

[http_service]
  internal_port = 7860
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[env]
  MODEL_CACHE_PATH = "/app/models"
  CONFIDENCE_THRESHOLD = "0.5"
  MAX_FRAMES_MEMORY = "10"

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024
"""

# Save fly.toml
with open('fly.toml', 'w') as f:
    f.write(fly_config)

# 6. Vercel configuration for serverless deployment
vercel_config = """{
  "version": 2,
  "builds": [
    {
      "src": "api_server.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api_server.py"
    }
  ],
  "env": {
    "MODEL_CACHE_PATH": "/tmp/models",
    "CONFIDENCE_THRESHOLD": "0.5",
    "MAX_FRAMES_MEMORY": "5"
  }
}"""

# Save vercel.json
with open('vercel.json', 'w') as f:
    f.write(vercel_config)

# 7. Environment template
env_template = """# Environment Variables Template
# Copy this file to .env and fill in your values

# Model Configuration
MODEL_CACHE_PATH=/app/models
CONFIDENCE_THRESHOLD=0.5
MAX_FRAMES_MEMORY=10

# Server Configuration  
HOST=0.0.0.0
PORT=8000

# Optional: API Keys for enhanced models
# HUGGINGFACE_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Performance Settings
MAX_WORKERS=1
TIMEOUT_KEEP_ALIVE=5

# Logging
LOG_LEVEL=INFO
"""

# Save .env.template
with open('.env.template', 'w') as f:
    f.write(env_template)

print("✅ Deployment configurations created!")
print("\nFiles created for different platforms:")
print("- railway.toml     - Railway deployment")
print("- render.yaml      - Render deployment") 
print("- Dockerfile       - Container deployment (HF Spaces, Fly.io)")
print("- README.md        - Hugging Face Spaces metadata")
print("- fly.toml         - Fly.io configuration")
print("- vercel.json      - Vercel serverless deployment")
print("- .env.template    - Environment variables template")