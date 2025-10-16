# Create a client example for using the deployed API
client_example_code = '''"""
Client Example for Enhanced Scene Understanding API
=================================================

This script demonstrates how to use the deployed FastAPI backend 
for scene understanding tasks. It includes examples for:
- Health checks
- Single frame processing
- Video analysis
- Real-time WebSocket streaming
"""

import requests
import json
import base64
import asyncio
import websockets
import cv2
from PIL import Image
import io
import time
from typing import Optional

class SceneUnderstandingClient:
    """Client for interacting with the Enhanced Scene Understanding API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the deployed API
                     Examples:
                     - Local: "http://localhost:8000"
                     - Railway: "https://your-project.up.railway.app"
                     - Render: "https://your-app.onrender.com"  
                     - HF Spaces: "https://username-app-name.hf.space"
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> dict:
        """Check if the API is healthy and ready"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Health check failed: {e}"}
    
    def get_system_info(self) -> dict:
        """Get detailed system information"""
        try:
            response = self.session.get(f"{self.base_url}/system-info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"System info request failed: {e}"}
    
    def process_image_file(self, image_path: str) -> dict:
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path, f, 'image/jpeg')}
                response = self.session.post(
                    f"{self.base_url}/process-frame",
                    files=files
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": f"Image processing failed: {e}"}
    
    def process_image_from_url(self, image_url: str) -> dict:
        """
        Download and process an image from URL
        
        Args:
            image_url: URL of the image to process
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Download image
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            # Process image
            files = {'file': ('image.jpg', img_response.content, 'image/jpeg')}
            response = self.session.post(
                f"{self.base_url}/process-frame",
                files=files
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"URL image processing failed: {e}"}
    
    def analyze_video_file(self, video_path: str, config: Optional[dict] = None) -> dict:
        """
        Analyze a video file
        
        Args:
            video_path: Path to the video file
            config: Optional processing configuration
            
        Returns:
            Dictionary containing complete video analysis
        """
        try:
            files = {'file': open(video_path, 'rb')}
            data = {}
            
            if config:
                data['config'] = json.dumps(config)
            
            response = self.session.post(
                f"{self.base_url}/analyze-video",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout for video processing
            )
            
            files['file'].close()
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": f"Video analysis failed: {e}"}
    
    async def stream_analysis(self, video_source=0, max_frames=100):
        """
        Real-time streaming analysis via WebSocket
        
        Args:
            video_source: Video source (0 for camera, or video file path)
            max_frames: Maximum number of frames to process
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        ws_url += '/stream-analysis'
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Initialize video capture
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    print(f"Error: Could not open video source {video_source}")
                    return
                
                frame_count = 0
                print(f"Starting real-time analysis... (max {max_frames} frames)")
                print("Press 'q' to quit\\n")
                
                try:
                    while frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert frame to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send frame to server
                        await websocket.send(json.dumps({
                            "frame": frame_b64,
                            "frame_id": frame_count
                        }))
                        
                        # Receive analysis result
                        try:
                            result = await asyncio.wait_for(
                                websocket.recv(), timeout=10.0
                            )
                            analysis = json.loads(result)
                            
                            if "error" in analysis:
                                print(f"Error: {analysis['error']}")
                            else:
                                self._display_stream_result(analysis, frame)
                            
                        except asyncio.TimeoutError:
                            print("Timeout waiting for analysis result")
                        
                        frame_count += 1
                        
                        # Small delay to prevent overwhelming the server
                        await asyncio.sleep(0.1)
                        
                        # Check for quit key (basic version)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    
        except Exception as e:
            print(f"WebSocket streaming error: {e}")
    
    def _display_stream_result(self, analysis: dict, frame):
        """Display streaming analysis results"""
        print(f"Frame {analysis.get('frame_id', 'Unknown')}:")
        print(f"  Scene: {analysis.get('scene_description', 'N/A')[:50]}...")
        print(f"  Confidence: {analysis.get('confidence_score', 0):.2f}")
        print(f"  Objects: {len(analysis.get('detections', []))}")
        
        # Draw results on frame for display
        if analysis.get('detections'):
            for det in analysis['detections']:
                bbox = det.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{det.get('class_name', 'Unknown')}: {det.get('confidence', 0):.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Real-time Scene Analysis', frame)
        print("-" * 60)

def demo_single_image():
    """Demo: Process a single image"""
    print("=== Single Image Processing Demo ===")
    
    # Initialize client
    client = SceneUnderstandingClient("http://localhost:8000")  # Change URL as needed
    
    # Health check first
    health = client.health_check()
    print(f"API Health: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("API is not healthy, skipping demo")
        return
    
    # Process image from URL (example)
    test_image_url = "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e"  # Food image
    print(f"Processing image from URL: {test_image_url}")
    
    result = client.process_image_from_url(test_image_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Scene Description: {result.get('scene_description')}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"Objects Detected: {len(result.get('detections', []))}")
        
        for i, detection in enumerate(result.get('detections', [])[:3]):  # Show top 3
            print(f"  {i+1}. {detection.get('class_name')}: {detection.get('confidence', 0):.2f}")

def demo_video_analysis():
    """Demo: Analyze a video file"""
    print("\\n=== Video Analysis Demo ===")
    
    client = SceneUnderstandingClient("http://localhost:8000")  # Change URL as needed
    
    # You would need to provide a video file path
    video_path = "sample_video.mp4"  # Replace with actual video path
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Skipping video analysis demo")
        return
    
    print(f"Analyzing video: {video_path}")
    print("This may take a while...")
    
    # Configure processing
    config = {
        "confidence_threshold": 0.6,
        "max_fps": 10,
        "enable_temporal": True
    }
    
    result = client.analyze_video_file(video_path, config)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Video Analysis Complete!")
        print(f"Total Frames: {result.get('total_frames', 0)}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"Average Confidence: {result.get('average_confidence', 0):.2f}")
        print(f"Summary: {result.get('summary', 'N/A')}")

async def demo_real_time_stream():
    """Demo: Real-time streaming analysis"""
    print("\\n=== Real-time Streaming Demo ===")
    
    client = SceneUnderstandingClient("http://localhost:8000")  # Change URL as needed
    
    print("Starting real-time camera analysis...")
    print("Make sure you have a camera connected!")
    
    try:
        await client.stream_analysis(video_source=0, max_frames=50)
    except Exception as e:
        print(f"Streaming demo failed: {e}")

def main():
    """Main demo function"""
    print("Enhanced Scene Understanding API Client Demo")
    print("=" * 50)
    
    # Import required modules
    import os
    
    # Demo 1: Single image processing
    demo_single_image()
    
    # Demo 2: Video analysis (if video file available)
    demo_video_analysis()
    
    # Demo 3: Real-time streaming (requires camera)
    print("\\nStarting real-time demo in 3 seconds...")
    print("Press Ctrl+C to skip")
    try:
        time.sleep(3)
        asyncio.run(demo_real_time_stream())
    except KeyboardInterrupt:
        print("\\nReal-time demo skipped")

if __name__ == "__main__":
    main()
'''

# Save the client example
with open('client_example.py', 'w') as f:
    f.write(client_example_code)

# Also create a simple test script
test_script = '''"""
Simple test script for the Enhanced Scene Understanding API
"""

import requests
import json

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""
    
    print(f"Testing API at: {base_url}")
    print("-" * 40)
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    print()
    
    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"✅ Health check: {response.status_code}")
        print(f"   Status: {health_data.get('status')}")
        print(f"   GPU Available: {health_data.get('gpu_available')}")
        print(f"   Models Loaded: {health_data.get('models_loaded')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    print()
    
    # Test 3: System info
    try:
        response = requests.get(f"{base_url}/system-info")
        info_data = response.json()
        print(f"✅ System info: {response.status_code}")
        print(f"   CPU Count: {info_data.get('system', {}).get('cpu_count')}")
        print(f"   Memory: {info_data.get('system', {}).get('memory_total')}GB")
        print(f"   PyTorch: {info_data.get('torch', {}).get('version')}")
    except Exception as e:
        print(f"❌ System info failed: {e}")
    
    print("\\n" + "=" * 40)
    print("API test completed!")
    
    # Instructions for further testing
    print("\\nFor full testing:")
    print("1. Use the client_example.py script")
    print("2. Visit the interactive docs at {base_url}/docs")
    print("3. Test with actual images and videos")

if __name__ == "__main__":
    import sys
    
    # Allow URL to be passed as command line argument
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(url)
'''

# Save the test script
with open('test_api.py', 'w') as f:
    f.write(test_script)

print("✅ Client examples and test scripts created!")
print("\nFiles created:")
print("- client_example.py    - Comprehensive client example")
print("- test_api.py         - Simple API testing script")
print("\nUsage examples:")
print("- python test_api.py                              # Test local API")
print("- python test_api.py https://your-app.railway.app # Test deployed API")
print("- python client_example.py                        # Run full demo")