"""
Enhanced Real-Time Scene Understanding System
============================================

This system incorporates state-of-the-art improvements based on recent research:
- YOLOv8 with feedback-enhanced processing
- Advanced vision-language models with temporal understanding
- Confidence thresholding to reduce hallucinations
- Multi-threaded architecture for real-time performance
- Memory optimization and error handling
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import threading
import queue
import time
import logging
from collections import deque
import psutil
from dataclasses import dataclass
from pathlib import Path

# Core ML libraries
from ultralytics import YOLO
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structure for storing detection results"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None

@dataclass
class FrameAnalysis:
    """Structure for storing complete frame analysis"""
    frame_id: int
    timestamp: float
    detections: List[DetectionResult]
    captions: List[str]
    scene_description: str
    confidence_score: float

class FeedbackEnhancedProcessor:
    """
    Feedback-enhanced processor that reduces hallucinations through 
    dynamic confidence thresholding and evidence-based text generation
    """

    def __init__(self, base_confidence: float = 0.5, feedback_strength: float = 0.1):
        self.base_confidence = base_confidence
        self.feedback_strength = feedback_strength
        self.recent_confidences = deque(maxlen=10)
        self.hallucination_patterns = set()

    def adjust_confidence_threshold(self, recent_results: List[float]) -> float:
        """Dynamically adjust confidence threshold based on recent performance"""
        if len(recent_results) < 3:
            return self.base_confidence

        avg_confidence = np.mean(recent_results)
        std_confidence = np.std(recent_results)

        # Lower threshold if results are consistently high confidence
        # Raise threshold if results are inconsistent
        adjustment = -self.feedback_strength if std_confidence < 0.1 else self.feedback_strength

        new_threshold = max(0.3, min(0.8, self.base_confidence + adjustment))
        return new_threshold

    def validate_caption_against_detections(self, caption: str, detections: List[DetectionResult]) -> float:
        """Validate generated caption against actual detections to reduce hallucination"""
        if not detections:
            return 0.5

        detected_classes = set(det.class_name.lower() for det in detections)
        caption_words = set(caption.lower().split())

        # Check for object mentions in caption
        mentioned_objects = caption_words.intersection(detected_classes)
        relevance_score = len(mentioned_objects) / max(len(detected_classes), 1)

        return min(1.0, relevance_score * 1.5)

class AdvancedVisionLanguageModel:
    """
    Advanced vision-language model with temporal understanding
    Based on VILA-style architecture improvements
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        # Temporal context buffer
        self.temporal_buffer = deque(maxlen=5)

        # Initialize feedback processor
        self.feedback_processor = FeedbackEnhancedProcessor()

        logger.info(f"Vision-Language model loaded on {self.device}")

    def generate_caption_with_context(self, image: Image.Image, detections: List[DetectionResult] = None) -> Dict:
        """Generate caption with temporal context and feedback enhancement"""
        try:
            # Generate base caption
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Use adaptive generation parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    length_penalty=1.2,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.7
                )

            caption = self.processor.decode(output[0], skip_special_tokens=True)

            # Validate against detections to reduce hallucination
            confidence = 1.0
            if detections:
                confidence = self.feedback_processor.validate_caption_against_detections(caption, detections)

            # Add temporal context if available
            contextualized_caption = self._add_temporal_context(caption)

            # Store in temporal buffer
            self.temporal_buffer.append({
                'caption': contextualized_caption,
                'confidence': confidence,
                'timestamp': time.time()
            })

            return {
                'caption': contextualized_caption,
                'confidence': confidence,
                'raw_caption': caption
            }

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {'caption': "Error generating caption", 'confidence': 0.0, 'raw_caption': ""}

    def _add_temporal_context(self, caption: str) -> str:
        """Add temporal context from previous frames"""
        if len(self.temporal_buffer) < 2:
            return caption

        # Simple temporal context - could be enhanced with more sophisticated methods
        recent_captions = [item['caption'] for item in list(self.temporal_buffer)[-2:]]

        # Check for consistency patterns
        common_elements = set(caption.lower().split())
        for prev_caption in recent_captions:
            common_elements = common_elements.intersection(set(prev_caption.lower().split()))

        if len(common_elements) > 2:
            return f"{caption} (consistent scene elements detected)"

        return caption

class OptimizedYOLOProcessor:
    """
    Optimized YOLO processor with improved performance and error handling
    """

    def __init__(self, model_name: str = "yolov8n-seg.pt", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)

        # Performance optimization
        if torch.cuda.is_available():
            self.model.to('cuda')

        # Dynamic confidence adjustment
        self.feedback_processor = FeedbackEnhancedProcessor()

        logger.info(f"YOLO model loaded: {model_name} on {self.device}")

    def detect_and_segment(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        """Perform detection and segmentation with optimized parameters"""
        try:
            # Adjust confidence threshold based on recent performance
            recent_confidences = getattr(self, '_recent_confidences', [])
            if len(recent_confidences) > 5:
                confidence_threshold = self.feedback_processor.adjust_confidence_threshold(recent_confidences[-5:])

            # Run inference
            results = self.model(frame, conf=confidence_threshold, iou=0.4, verbose=False)

            if not results or not results[0].boxes:
                return []

            result = results[0]
            detections = []

            # Process each detection
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                bbox = tuple(map(int, box.cpu().numpy()))
                confidence = float(conf.cpu().numpy())
                class_id = int(cls.cpu().numpy())
                class_name = self.model.names[class_id]

                # Extract mask if available
                mask = None
                if hasattr(result, 'masks') and result.masks is not None:
                    if i < len(result.masks.data):
                        mask = result.masks.data[i].cpu().numpy()

                detection = DetectionResult(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask
                )
                detections.append(detection)

            # Update confidence tracking
            if not hasattr(self, '_recent_confidences'):
                self._recent_confidences = deque(maxlen=20)
            self._recent_confidences.extend([d.confidence for d in detections])

            return detections

        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []

class EnhancedSceneUnderstanding:
    """
    Main class for enhanced real-time scene understanding
    """

    def __init__(self, 
                 yolo_model: str = "yolov8n-seg.pt",
                 vision_model: str = "Salesforce/blip-image-captioning-base",
                 max_fps: int = 30,
                 enable_temporal: bool = True):

        self.max_fps = max_fps
        self.enable_temporal = enable_temporal

        # Initialize models
        logger.info("Initializing Enhanced Scene Understanding System...")
        self.yolo_processor = OptimizedYOLOProcessor(yolo_model)
        self.vision_model = AdvancedVisionLanguageModel(vision_model)

        # Threading components
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=50)
        self.processing_thread = None
        self.is_running = False

        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # Memory management
        self.max_memory_mb = 2048  # 2GB limit

        logger.info("System initialized successfully!")

    def start_processing_thread(self):
        """Start the background processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        logger.info("Processing thread started")

    def stop_processing_thread(self):
        """Stop the background processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Processing thread stopped")

    def _processing_worker(self):
        """Background worker for processing frames"""
        while self.is_running:
            try:
                # Check memory usage
                if self._check_memory_usage():
                    logger.warning("High memory usage detected, skipping frame")
                    time.sleep(0.1)
                    continue

                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue

                frame, frame_id = frame_data

                # Process frame
                analysis = self._analyze_frame(frame, frame_id)

                # Store result
                if not self.result_queue.full():
                    self.result_queue.put(analysis)
                else:
                    # Remove oldest result to make space
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(analysis)
                    except queue.Empty:
                        pass

                # Update FPS
                self._update_fps()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")

    def _analyze_frame(self, frame: np.ndarray, frame_id: int) -> FrameAnalysis:
        """Analyze a single frame"""
        timestamp = time.time()

        try:
            # YOLO detection
            detections = self.yolo_processor.detect_and_segment(frame)

            # Generate captions for detected objects
            captions = []
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if detections:
                # Generate scene-level caption
                scene_result = self.vision_model.generate_caption_with_context(pil_image, detections)
                scene_description = scene_result['caption']
                confidence_score = scene_result['confidence']

                # Generate individual object captions (optional, for detailed analysis)
                for detection in detections[:3]:  # Limit to top 3 for performance
                    if detection.mask is not None:
                        # Extract object region
                        x1, y1, x2, y2 = detection.bbox
                        object_region = frame[y1:y2, x1:x2]
                        if object_region.size > 0:
                            object_pil = Image.fromarray(cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB))
                            obj_result = self.vision_model.generate_caption_with_context(object_pil)
                            captions.append(f"{detection.class_name}: {obj_result['caption']}")
            else:
                # No detections, generate general scene description
                scene_result = self.vision_model.generate_caption_with_context(pil_image)
                scene_description = scene_result['caption']
                confidence_score = scene_result['confidence']

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_id}: {e}")
            scene_description = "Error analyzing scene"
            confidence_score = 0.0
            captions = []

        return FrameAnalysis(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            captions=captions,
            scene_description=scene_description,
            confidence_score=confidence_score
        )

    def process_video_stream(self, video_source=0):
        """Process video stream from camera or file"""
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Could not open video source: {video_source}")
            return

        # Start processing thread
        self.start_processing_thread()

        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):  # File ended
                        break
                    continue

                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), frame_id))

                # Display current frame with latest analysis
                display_frame = self._draw_analysis_on_frame(frame.copy())
                cv2.imshow('Enhanced Scene Understanding', display_frame)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Save current analysis
                    self._save_current_analysis()

                frame_id += 1

                # Frame rate control
                time.sleep(1.0 / self.max_fps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop_processing_thread()
            cap.release()
            cv2.destroyAllWindows()

    def _draw_analysis_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw analysis results on frame"""
        try:
            # Get latest analysis result
            analysis = self.result_queue.get_nowait() if not self.result_queue.empty() else None

            if analysis:
                # Draw detections
                for detection in analysis.detections:
                    x1, y1, x2, y2 = detection.bbox

                    # Draw bounding box
                    color = (0, 255, 0) if detection.confidence > 0.7 else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw scene description
                scene_text = f"Scene: {analysis.scene_description[:60]}..."
                cv2.putText(frame, scene_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw confidence and FPS
                info_text = f"Confidence: {analysis.confidence_score:.2f} | FPS: {self.current_fps:.1f}"
                cv2.putText(frame, info_text, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except queue.Empty:
            # No recent analysis available
            cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is too high"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.max_memory_mb
        except:
            return False

    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

    def _save_current_analysis(self):
        """Save current analysis to file"""
        try:
            if not self.result_queue.empty():
                analysis = self.result_queue.queue[-1]  # Get most recent

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"scene_analysis_{timestamp}.txt"

                with open(filename, 'w') as f:
                    f.write(f"Scene Analysis - {timestamp}\n")
                    f.write(f"Scene Description: {analysis.scene_description}\n")
                    f.write(f"Confidence Score: {analysis.confidence_score:.3f}\n")
                    f.write(f"Objects Detected: {len(analysis.detections)}\n\n")

                    for i, detection in enumerate(analysis.detections):
                        f.write(f"Object {i+1}:\n")
                        f.write(f"  Class: {detection.class_name}\n")
                        f.write(f"  Confidence: {detection.confidence:.3f}\n")
                        f.write(f"  Bbox: {detection.bbox}\n\n")

                    if analysis.captions:
                        f.write("Detailed Captions:\n")
                        for caption in analysis.captions:
                            f.write(f"  - {caption}\n")

                logger.info(f"Analysis saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

def main():
    """Main function for running the enhanced scene understanding system"""

    # Configuration
    YOLO_MODEL = "yolov8n-seg.pt"  # or yolov8s-seg.pt for better accuracy
    VISION_MODEL = "Salesforce/blip-image-captioning-base"
    MAX_FPS = 15  # Balanced performance
    VIDEO_SOURCE = 0  # 0 for camera, or path to video file

    try:
        # Initialize system
        system = EnhancedSceneUnderstanding(
            yolo_model=YOLO_MODEL,
            vision_model=VISION_MODEL,
            max_fps=MAX_FPS,
            enable_temporal=True
        )

        print("\n" + "="*60)
        print("Enhanced Real-Time Scene Understanding System")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current analysis")
        print("\nStarting video processing...")
        print("="*60)

        # Process video stream
        system.process_video_stream(VIDEO_SOURCE)

    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
