from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8l.pt")

try:
    # Use the model to make a prediction on the webcam feed (source=0)
    # Loop to continuously capture frames and make predictions
    while True:
        result = model.predict(source=0, show=True)
        print(result)
except KeyboardInterrupt:
    # Handle the keyboard interrupt to stop the script
    print("Stopping the script...")
finally:
    # Release resources if needed
    cv2.destroyAllWindows()
    print("Resources released and script stopped.")
