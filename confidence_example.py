from ultralytics import YOLO
import cv2

# Load YOLOv8 model (segmentation model)
model = YOLO("yolov8n-seg.pt")  # Replace with the path to your model or use a pre-trained one.

# Load an image
image_path = "your_image.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Define a confidence threshold
CONF_THRESHOLD = 0.8

# Process results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    probs = result.boxes.conf.cpu().numpy()  # Confidence scores
    
    # Iterate through detected objects
    for i, prob in enumerate(probs):
        if prob > CONF_THRESHOLD:
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            print(f"Object {i+1}:")
            print(f"  Confidence Score: {prob}")
            print(f"  Bounding Box: {box}")
            
            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(image, f"{prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
