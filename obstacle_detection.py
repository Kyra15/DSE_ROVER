import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy library for numerical operations

def detect_vertical_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect edges using the Canny edge detector
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines using the probabilistic Hough line transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:  # Check if any lines are detected
        for line in lines:  # Iterate through each detected line
            for x1, y1, x2, y2 in line:  # Extract the coordinates of the line endpoints
                # Calculate the angle of the line in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Check if the line is approximately vertical (angle close to 90 or -90 degrees)
                if abs(angle) > 80 and abs(angle) < 100:
                    # Draw a rectangle around the detected vertical line
                    cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 255, 0), 2)
                    # Put a label "Obstacle" near the detected line
                    cv2.putText(frame, 'Obstacle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame  # Return the frame with detected lines highlighted

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
while True:  # Continuously capture frames from the webcam
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:  # If no frame is captured, break the loop
        break
    # Process the frame to detect vertical lines
    processed_frame = detect_vertical_lines(frame)
    # Display the processed frame
    cv2.imshow('Frame', processed_frame)
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
