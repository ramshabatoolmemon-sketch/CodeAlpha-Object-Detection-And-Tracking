from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    # Object detection + tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.5
    )

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection and Tracking", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()