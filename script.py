# Create enhanced requirements.txt with optimized dependencies
requirements_content = """# Core ML/AI libraries
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
transformers>=4.30.0
pillow>=9.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# FastAPI and web framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
websockets>=11.0

# Additional ML models and processing
accelerate>=0.21.0
safetensors>=0.3.0
tokenizers>=0.13.0

# Image and video processing
imageio>=2.31.0
moviepy>=1.0.0
scikit-image>=0.21.0

# Utility and optimization
psutil>=5.9.0
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0

# Optional: For better performance
# onnxruntime  # Uncomment for ONNX inference
# tensorrt     # Uncomment for TensorRT optimization (NVIDIA GPUs)

# Development and deployment
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.7.0"""

# Save requirements.txt
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ Enhanced requirements.txt created successfully!")
print("\nKey improvements in dependencies:")
print("- Latest stable versions for better performance")
print("- FastAPI ecosystem for API deployment")
print("- WebSocket support for real-time streaming")
print("- Optimized image/video processing libraries")
print("- Development tools for deployment")