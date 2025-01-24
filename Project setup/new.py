import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO  # YOLOv8 library
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


class YOLOv8Model:
    def __init__(self):
        """
        Initializes the YOLOv8 model for segmentation and detection, with GPU support if available.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.setup_yolov8()

    def setup_yolov8(self):
        """
        Sets up the YOLOv8 model for segmentation and detection, with GPU support if available.
        Returns:
            model (YOLO): Pre-trained YOLOv8 model.
        """
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8n-seg.pt')  # YOLOv8 segmentation model
        model.to(self.device)  # Move the model to GPU if available
        print(f"YOLOv8 model loaded successfully and moved to {self.device}.")
        return model


class BLIPModel:
    def __init__(self):
        """
        Initializes the BLIP model and processor, with GPU support if available.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor, self.model = self.setup_blip()

    def setup_blip(self):
        """
        Sets up the BLIP model and processor, with GPU support if available.
        Returns:
            processor (BlipProcessor): Preprocessing pipeline for images.
            model (BlipForConditionalGeneration): BLIP model for caption generation.
        """
        print("Loading BLIP model and processor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(self.device)  # Move the model to GPU if available
        print(f"BLIP model loaded successfully and moved to {self.device}.")
        return processor, model

    def generate_caption(self, image):
        """
        Generates a caption for a given image using the BLIP model, with GPU support if available.

        Args:
            image (PIL.Image): Image for which the caption is to be generated.

        Returns:
            str: Generated caption.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # Move inputs to GPU if available
        output = self.model.generate(**inputs, max_new_tokens=128)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption


class VideoProcessor:
    def __init__(self, input_video_path, output_video_path):
        """
        Initializes the VideoProcessor with input and output video paths.

        Args:
            input_video_path (str): Path to the input video file.
            output_video_path (str): Path to the output video file.
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.yolo_model = YOLOv8Model()
        self.blip_model = BLIPModel()

    def draw_bounding_boxes_and_labels(self, frame, results, captions):
        """
        Draws bounding boxes, class labels, and captions on the frame.

        Args:
            frame (numpy.ndarray): The frame to annotate.
            results (Results): YOLOv8 detection results.
            captions (list): Captions generated for detected objects.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        for idx, (box, cls, caption) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, captions)):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = results[0].names[int(cls)]  # Class name from YOLOv8
            label = f"{class_name}: {caption}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            # Add label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def process_yolo_and_generate_captions(self, results, frame):
        """
        Processes YOLOv8 results and generates captions for detected objects.

        Args:
            results (Results): YOLOv8 detection results.
            frame (numpy.ndarray): The current video frame.

        Returns:
            list: Captions for each detected object.
        """
        captions = []
        frame_height, frame_width = frame.shape[:2]

        for box, mask in zip(results[0].boxes.xyxy, results[0].masks.data):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Ensure bounding box coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)

            # Convert mask to uint8 and resize to match ROI size
            mask = (mask.cpu().numpy() * 255).astype("uint8")
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:  # Ensure the ROI is valid
                roi_height, roi_width = roi.shape[:2]
                resized_mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

                # Apply the resized mask to the ROI
                if resized_mask.shape[:2] == roi.shape[:2]:  # Ensure sizes match
                    masked_roi = cv2.bitwise_and(roi, roi, mask=resized_mask)
                    pil_image = Image.fromarray(cv2.cvtColor(masked_roi, cv2.COLOR_BGR2RGB))
                    captions.append(self.blip_model.generate_caption(pil_image))
                else:
                    captions.append("Invalid mask and ROI size mismatch")
            else:
                captions.append("No valid ROI detected")

        return captions

    def process_video(self):
        """
        Processes the input video, performs segmentation and detection using YOLOv8,
        generates captions using BLIP, and saves the annotated video.
        """
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_video_path}.")
            return

        # Get the video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        print("Processing video... This may take some time.")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading the frame.")
                break

            # Perform segmentation and detection using YOLOv8
            frame_tensor = torch.from_numpy(frame).to(self.yolo_model.device)  # Move frame to GPU if available
            results = self.yolo_model.model(frame_tensor, conf=0.5, iou=0.4)

            # Generate captions for each masked image
            captions = self.process_yolo_and_generate_captions(results, frame)

            # Annotate the frame with bounding boxes and captions
            annotated_frame = self.draw_bounding_boxes_and_labels(frame, results, captions)

            # Write the annotated frame to the output video
            out.write(annotated_frame)
            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames}")

        cap.release()
        out.release()
        print(f"Processed video saved to: {self.output_video_path}")


class Summarizer:
    def __init__(self, model_name="allenai/led-large-16384"):
        """
        Initializes the Summarizer with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_length=500, min_length=100):
        """
        Generates a summary of the given text.

        Args:
            text (str): Text to summarize.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.

        Returns:
            str: Generated summary.
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=16384, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


if __name__ == "__main__":
    input_video_path = "classroom_video.mp4"  # Path to your video
    output_video_path = "processed_video.mp4"  # Output file name
    captions_file = "captions_log.txt"
    summary_file = "video_summary_led.txt"

    # Clear captions log
    open(captions_file, "w").close()

    video_processor = VideoProcessor(input_video_path, output_video_path)
    summarizer = Summarizer()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    print("Processing video...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection
        results = video_processor.yolo_model.model(frame, conf=0.5, iou=0.4)

        # Generate captions
        captions = video_processor.process_yolo_and_generate_captions(results, frame)

        # Save captions to file
        with open(captions_file, "a") as f:
            f.write(f"Frame {frame_count}:\n")
            for idx, caption in enumerate(captions):
                f.write(f"  Object {idx + 1}: {caption}\n")
            f.write("\n")

        # Annotate frame
        annotated_frame = video_processor.draw_bounding_boxes_and_labels(frame, results, captions)
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    print("Generating video summary...")
    with open(captions_file, "r") as f:
        captions_content = f.read()
    summary = summarizer.summarize(captions_content)
    with open(summary_file, "w") as f:
        f.write(summary)
    print("Video Summary:")
    print(summary)
