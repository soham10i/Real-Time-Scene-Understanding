import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO  # YOLOv8 library
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


def setup_yolov8():
    """Sets up the YOLOv8 model for segmentation and detection."""
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n-seg.pt')  # YOLOv8 segmentation model
    if torch.cuda.is_available():
        model.to('cuda')  # Use GPU if available
        print("YOLOv8 model loaded successfully on GPU.")
    else:
        print("YOLOv8 model loaded successfully on CPU.")
    return model


def setup_blip():
    """Sets up the BLIP model and processor."""
    print("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    if torch.cuda.is_available():
        model.to('cuda')  # Use GPU if available
        print("BLIP model loaded successfully on GPU.")
    else:
        print("BLIP model loaded successfully on CPU.")
    return processor, model


def generate_caption(processor, model, image):
    """Generates a caption for a given image using the BLIP model."""
    inputs = processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    output = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def draw_bounding_boxes_and_labels(frame, results, captions):
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


def process_yolo_and_generate_captions(results, frame, processor, blip_model):
    """
    Processes YOLOv8 results and generates captions for detected objects.

    Args:
        results (Results): YOLOv8 detection results.
        frame (numpy.ndarray): The current video frame.
        processor (BlipProcessor): BLIP processor.
        blip_model (BlipForConditionalGeneration): BLIP model.

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
                captions.append(generate_caption(processor, blip_model, pil_image))
            else:
                captions.append("Invalid mask and ROI size mismatch")
        else:
            captions.append("No valid ROI detected")

    return captions



def summarize_with_longformer(captions_file, output_summary_file="video_summary_led.txt"):
    """
    Summarizes the content of the video using captions from the captions log file, leveraging Longformer (LED).

    Args:
        captions_file (str): Path to the captions log file.
        output_summary_file (str): Path to save the generated summary.

    Returns:
        str: Generated summary of the video.
    """
    # Load the captions log
    with open(captions_file, "r") as f:
        captions_content = f.readlines()

    # Extract unique lines and remove frame labels
    captions = []
    for line in captions_content:
        if "Object" in line:  # Skip frame headers and focus on object captions
            caption = line.split(": ", 1)[-1].strip()
            if caption not in captions:
                captions.append(caption)

    # Combine unique captions into one input for summarization
    captions_text = " ".join(captions)

    # Load Longformer (LED) tokenizer and model
    model_name = "allenai/led-large-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize and truncate input to fit Longformer's token limit
    inputs = tokenizer(captions_text, return_tensors="pt", max_length=16384, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs.input_ids, max_length=500, min_length=100, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Save the summary to a file
    with open(output_summary_file, "w") as f:
        f.write(summary)

    print(f"Summary saved to: {output_summary_file}")
    return summary


def main():
    yolo_model = setup_yolov8()
    processor, blip_model = setup_blip()
    captions_file = "captions_log.txt"
    summary_file = "video_summary_led.txt"

    # Clear captions log
    open(captions_file, "w").close()

    cap = cv2.VideoCapture("classroom_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("processed_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection and caption generation
        results = yolo_model(frame, conf=0.5, iou=0.4)
        captions = process_yolo_and_generate_captions(results, frame, processor, blip_model)

        # Save captions to file
        with open(captions_file, "a") as f:
            f.write(f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}:\n")
            for idx, caption in enumerate(captions):
                f.write(f"  Object {idx + 1}: {caption}\n")
            f.write("\n")

    cap.release()
    out.release()

    print("Generating video summary...")
    summary = summarize_with_longformer(captions_file, summary_file)
    print("Video Summary:")
    print(summary)



if __name__ == "__main__":
    main()
