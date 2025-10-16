import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO  # YOLOv8 library
from transformers import BlipProcessor, BlipForConditionalGeneration

def setup_yolov8():
    """
    Sets up the YOLOv8 model for segmentation and detection.
    Returns:
        model (YOLO): Pre-trained YOLOv8 model.
    """
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n-seg.pt')  # YOLOv8 segmentation model
    print("YOLOv8 model loaded successfully.")
    return model

def setup_blip():
    """
    Sets up the BLIP model and processor.
    Returns:
        processor (BlipProcessor): Preprocessing pipeline for images.
        model (BlipForConditionalGeneration): BLIP model for caption generation.
    """
    print("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("BLIP model loaded successfully.")
    return processor, model

def generate_caption(processor, model, image):
    """
    Generates a caption for a given image using the BLIP model.
    
    Args:
        processor (BlipProcessor): BLIP processor for input preparation.
        model (BlipForConditionalGeneration): BLIP model for caption generation.
        image (PIL.Image): Image for which the caption is to be generated.

    Returns:
        str: Generated caption.
    """
    # Preprocess and generate the caption
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def process_yolo_output(results, frame, output_folder, frame_count):
    """
    Processes YOLOv8 results to extract and save masked images.
    
    Args:
        results (Results): YOLOv8 output results.
        frame (numpy.ndarray): Original frame.
        output_folder (str): Path to save processed images.
        frame_count (int): Frame count for file naming.

    Returns:
        List[PIL.Image]: List of masked images as PIL Image objects.
    """
    # Access the first result from the list
    result = results[0]

    # Extract bounding boxes and masks
    masked_images = []
    for idx, (box, mask) in enumerate(zip(result.boxes.xyxy, result.masks.data)):

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, box.tolist())

        # Convert mask to binary and apply to the frame
        mask = (mask.cpu().numpy() * 255).astype("uint8")  # Convert mask to binary
        roi = frame[y1:y2, x1:x2]
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask[y1:y2, x1:x2])

        # Save masked image
        filename = os.path.join(output_folder, f"frame_{frame_count}_object_{idx}.jpg")
        cv2.imwrite(filename, masked_roi)
        print(f"Saved masked object: {filename}")

        # Convert to PIL Image and append to the list
        masked_images.append(Image.fromarray(cv2.cvtColor(masked_roi, cv2.COLOR_BGR2RGB)))

    return masked_images


def main():
    
    yolo_model = setup_yolov8()
    processor, blip_model = setup_blip()

    output_folder = "processed_frames"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the application.")
    frame_count = 0  # To track frame count for file naming
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform segmentation and detection using YOLOv8
        results = yolo_model(frame, conf=0.5, iou=0.4)

        # Process YOLOv8 output to extract masked images
        masked_images = process_yolo_output(results, frame, output_folder, frame_count)

        # Generate captions for each masked image
        for idx, masked_image in enumerate(masked_images):
            caption = generate_caption(processor, blip_model, masked_image)
            print(f"Object {idx + 1}: {caption}")

        # Display the frame with bounding boxes and segmentation masks
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Segmentation and Detection", annotated_frame)
        # print(results[0])
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed frames and masked images saved in folder: {output_folder}")
    print("Application closed.")

if __name__ == "__main__":
    main()
