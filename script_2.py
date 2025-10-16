# Create FastAPI backend server
fastapi_server_code = '''"""
FastAPI Backend for Enhanced Real-Time Scene Understanding
=========================================================

This FastAPI server provides RESTful API endpoints for the enhanced scene understanding system.
Supports video upload, real-time streaming, and batch processing capabilities.

Features:
- Video file upload and analysis
- WebSocket real-time streaming
- Single frame processing
- Health checks and system monitoring
- CORS support for web integration
- Async processing for better performance
"""

import os
import io
import asyncio
import logging
from typing import List, Dict, Optional, Union
import uuid
from pathlib import Path
import tempfile
import base64
import json
from datetime import datetime

# FastAPI imports
from fastapi import (
    FastAPI, 
    File, 
    UploadFile, 
    WebSocket, 
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Core libraries
import cv2
import numpy as np
from PIL import Image
import torch

# Import our enhanced system (assuming it's in the same directory)
try:
    from enhanced_scene_understanding import (
        EnhancedSceneUnderstanding,
        FrameAnalysis,
        DetectionResult
    )
except ImportError:
    # Fallback minimal implementation for deployment
    logging.warning("Enhanced scene understanding module not found, using fallback")
    EnhancedSceneUnderstanding = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ProcessingConfig(BaseModel):
    """Configuration for processing parameters"""
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    max_fps: int = Field(default=15, ge=1, le=60)
    enable_temporal: bool = Field(default=True)
    yolo_model: str = Field(default="yolov8n-seg.pt")
    vision_model: str = Field(default="Salesforce/blip-image-captioning-base")

class AnalysisResult(BaseModel):
    """Result model for frame analysis"""
    frame_id: int
    timestamp: float
    scene_description: str
    confidence_score: float
    detections: List[Dict]
    captions: List[str]
    processing_time: float

class VideoAnalysisResult(BaseModel):
    """Result model for complete video analysis"""
    video_id: str
    total_frames: int
    processing_time: float
    summary: str
    frames_analyzed: List[AnalysisResult]
    average_confidence: float

class HealthStatus(BaseModel):
    """System health status model"""
    status: str
    timestamp: str
    system_info: Dict
    gpu_available: bool
    models_loaded: bool

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Scene Understanding API",
    description="Advanced real-time scene understanding with YOLO and Vision-Language models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
scene_understanding_system: Optional[EnhancedSceneUnderstanding] = None
active_websockets: List[WebSocket] = []
processing_tasks: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global scene_understanding_system
    
    try:
        logger.info("Initializing Enhanced Scene Understanding System...")
        
        # Check if enhanced system is available
        if EnhancedSceneUnderstanding is not None:
            scene_understanding_system = EnhancedSceneUnderstanding(
                yolo_model="yolov8n-seg.pt",
                vision_model="Salesforce/blip-image-captioning-base",
                max_fps=15,
                enable_temporal=True
            )
            logger.info("System initialized successfully!")
        else:
            logger.warning("Enhanced system not available - running in limited mode")
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global scene_understanding_system
    
    if scene_understanding_system:
        scene_understanding_system.stop_processing_thread()
        
    # Close all active WebSocket connections
    for websocket in active_websockets:
        try:
            await websocket.close()
        except:
            pass

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Scene Understanding API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze_video": "/analyze-video",
            "process_frame": "/process-frame",
            "stream_analysis": "/stream-analysis",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    import psutil
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent if os.path.exists('/') else 0
    }
    
    return HealthStatus(
        status="healthy" if scene_understanding_system else "limited",
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        gpu_available=torch.cuda.is_available(),
        models_loaded=scene_understanding_system is not None
    )

@app.post("/analyze-video", response_model=VideoAnalysisResult)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: ProcessingConfig = ProcessingConfig()
):
    """
    Analyze uploaded video file
    """
    if not scene_understanding_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scene understanding system not available"
        )
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    video_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Process video
        start_time = datetime.now()
        results = await process_video_file(temp_path, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up temp file
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Calculate summary statistics
        if results:
            avg_confidence = sum(r['confidence_score'] for r in results) / len(results)
            summary = generate_video_summary(results)
        else:
            avg_confidence = 0.0
            summary = "No analysis results generated"
        
        return VideoAnalysisResult(
            video_id=video_id,
            total_frames=len(results),
            processing_time=processing_time,
            summary=summary,
            frames_analyzed=[AnalysisResult(**result) for result in results],
            average_confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {str(e)}"
        )

@app.post("/process-frame", response_model=AnalysisResult)
async def process_frame(
    file: UploadFile = File(...),
    config: ProcessingConfig = ProcessingConfig()
):
    """
    Process a single image frame
    """
    if not scene_understanding_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scene understanding system not available"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Read and process image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze frame
        start_time = datetime.now()
        analysis = scene_understanding_system._analyze_frame(frame, 0)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to API response format
        detections_dict = []
        for det in analysis.detections:
            detections_dict.append({
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name
            })
        
        return AnalysisResult(
            frame_id=analysis.frame_id,
            timestamp=analysis.timestamp,
            scene_description=analysis.scene_description,
            confidence_score=analysis.confidence_score,
            detections=detections_dict,
            captions=analysis.captions,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame processing failed: {str(e)}"
        )

@app.websocket("/stream-analysis")
async def websocket_stream_analysis(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame analysis
    Client sends base64 encoded frames, server responds with analysis
    """
    await websocket.accept()
    active_websockets.append(websocket)
    
    if not scene_understanding_system:
        await websocket.send_json({
            "error": "Scene understanding system not available"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if "frame" not in data:
                await websocket.send_json({"error": "No frame data provided"})
                continue
            
            # Decode base64 frame
            try:
                frame_data = base64.b64decode(data["frame"])
                image = Image.open(io.BytesIO(frame_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid frame data: {e}"})
                continue
            
            # Process frame
            try:
                analysis = scene_understanding_system._analyze_frame(frame, data.get("frame_id", 0))
                
                # Convert to JSON serializable format
                response = {
                    "frame_id": analysis.frame_id,
                    "timestamp": analysis.timestamp,
                    "scene_description": analysis.scene_description,
                    "confidence_score": analysis.confidence_score,
                    "detections": [
                        {
                            "bbox": det.bbox,
                            "confidence": det.confidence,
                            "class_id": det.class_id,
                            "class_name": det.class_name
                        }
                        for det in analysis.detections
                    ],
                    "captions": analysis.captions
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                await websocket.send_json({"error": f"Processing failed: {e}"})
                
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

@app.get("/system-info")
async def get_system_info():
    """Get detailed system information"""
    import psutil
    
    info = {
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
            "disk_total": psutil.disk_usage('/').total // (1024**3) if os.path.exists('/') else 0  # GB
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "models": {
            "system_loaded": scene_understanding_system is not None,
            "active_websockets": len(active_websockets)
        }
    }
    
    return info

# Utility functions
async def process_video_file(video_path: str, config: ProcessingConfig) -> List[Dict]:
    """Process video file and return analysis results"""
    results = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_id = 0
        sample_rate = max(1, int(cap.get(cv2.CAP_PROP_FPS) / config.max_fps))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on desired FPS
            if frame_id % sample_rate == 0:
                analysis = scene_understanding_system._analyze_frame(frame, frame_id)
                
                result = {
                    "frame_id": analysis.frame_id,
                    "timestamp": analysis.timestamp,
                    "scene_description": analysis.scene_description,
                    "confidence_score": analysis.confidence_score,
                    "detections": [
                        {
                            "bbox": det.bbox,
                            "confidence": det.confidence,
                            "class_id": det.class_id,
                            "class_name": det.class_name
                        }
                        for det in analysis.detections
                    ],
                    "captions": analysis.captions,
                    "processing_time": 0.0  # Would need to measure actual processing time
                }
                results.append(result)
            
            frame_id += 1
        
        cap.release()
        
    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        raise
    
    return results

def generate_video_summary(results: List[Dict]) -> str:
    """Generate a summary of video analysis results"""
    if not results:
        return "No frames analyzed"
    
    # Collect scene descriptions
    descriptions = [r["scene_description"] for r in results if r["scene_description"]]
    
    # Simple summary generation (could be enhanced with actual summarization models)
    if descriptions:
        # Find most common elements
        all_words = " ".join(descriptions).lower().split()
        word_freq = {}
        for word in all_words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary = f"Video contains {len(results)} analyzed frames. "
        if top_words:
            common_elements = ", ".join([word for word, _ in top_words[:5]])
            summary += f"Common scene elements: {common_elements}. "
        
        avg_confidence = sum(r["confidence_score"] for r in results) / len(results)
        summary += f"Average confidence: {avg_confidence:.2f}"
        
        return summary
    
    return "Video analysis completed with limited results"

async def cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up temp file: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting FastAPI server on {HOST}:{PORT}")
    
    uvicorn.run(
        "api_server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
'''

# Save the FastAPI server
with open('api_server.py', 'w') as f:
    f.write(fastapi_server_code)

print("✅ FastAPI Backend Server created!")
print("\nAPI Endpoints implemented:")
print("- GET /                    - Root endpoint with API info")
print("- GET /health              - System health check") 
print("- POST /analyze-video      - Upload and analyze video files")
print("- POST /process-frame      - Process single image frame")
print("- WebSocket /stream-analysis - Real-time frame analysis")
print("- GET /system-info         - Detailed system information")
print("- GET /docs               - Interactive API documentation")