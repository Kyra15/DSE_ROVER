import io
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk   
from datetime import datetime
import logging
import pytz
import threading

# Root Menu for user registration and login
root = tk.Tk()
root.title("User Login")

# Set up logging with US/Eastern timezone
eastern = pytz.timezone('US/Eastern')

#Set up global variables for video streaming control
stop_video_raw = threading.Event()
stop_video_processed = threading.Event()

# Initialize webcam
webcam = cv.VideoCapture("http://192.168.1.14:5000/video_feed")
webcam_lock = threading.Lock()

# Updat the lable for log file
def update_label(image, label):
    photo = ImageTk.PhotoImage(image=image)
    label.config(image=photo)
    label.image = photo  # Keep a reference

#Create a log file to track activities
# Set up logging
logging.basicConfig(
    filename='activity_log.txt',
    level=logging.INFO,
    format='%(message)s',
)

# Function for the log information including date and time and message
def log_activity(message):
  current_time = datetime.now(eastern).strftime('%Y-%m-%d %I:%M:%S %p')
  logging.info(f"{current_time} - {message}")


def getCanny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def getSegment(frame):
    height, width = frame.shape[:2]
    # Define the vertices of the trapezoid
    # Adjust these points based on your camera setup
    lower_left = [width*0.01, height]
    lower_right = [width*0.99, height]
    upper_left = [width*0.01, height*0.5]
    upper_right = [width*0.99, height*0.5]
    polygons = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)
    segment = cv.bitwise_and(frame, mask)
    return segment, mask 

def generateLines(frame, lines):
    left = []
    right = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))

    # Add handling if all slopes are zero or near zero
    if not left and not right:
        return None

    if left:
        left_avg = np.average(left, axis=0)
        left_line = generateCoordinates(frame, left_avg)
    else:
        left_line = None  # or set a default value

    if right:
        right_avg = np.average(right, axis=0)
        right_line = generateCoordinates(frame, right_avg)
    else:
        right_line = None  # or set a default value

    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else None

def generateCoordinates(frame, parameters):
    slope, intercept = parameters
    y1 = frame.shape[0]  # Bottom of the frame
    y2 = int(y1 * 0.6)   # Extend the line higher up in the frame

    # Check for zero or near-zero slope to avoid division by zero
    if np.isclose(slope, 0):
        x1 = x2 = int(intercept)
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


# Function to visualize the lines on the frame, including the centerline
def showLines(frame, lines):
    try:
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Draw lane lines in green

            # Calculate and draw the centerline
            left_line, right_line = lines
            center_line = (left_line + right_line) // 2
            x1, y1, x2, y2 = center_line
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Draw centerline in red

    except Exception as e:
        print(f"Error in showLines: {e}")
        # Optionally, log this error or take other appropriate actions

    return frame

def load_video_raw(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Camera loaded")
    #log_activity(f"{userName} clicked load camera button.")
    video_label2 = tk.Label(frame)
    video_label2.grid(row=2, column=0, columnspan=2)

    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
    else:
        while not stop_event.is_set():
            with webcam_lock:
               ret, frame = webcam.read()
            if ret:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb_frame = cv.resize(rgb_frame, (256, 256))
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                video_label2.config(image=photo)
                video_label2.image = photo  # Keep a reference
                root.after(0, update_label, Image.fromarray(rgb_frame), video_label2)

def load_video_processed(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Overlay Loaded")
    #log_activity(f"{userName} clicked load overlay button.")
    video_label1 = tk.Label(frame)
    video_label1.grid(row=2, column=0, columnspan=2)
    max_attempts = 5  # Maximum number of attempts to read from the webcam
    attempts = 0  # Current number of attempts

    if not webcam or not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return
    else:
        while not stop_event.is_set():
            with webcam_lock:
                try:
                    ret, original_frame = webcam.read()
                    if not ret:
                        raise ValueError("Unable to read from webcam")
                    attempts = 0
                except Exception as e:
                    print(f"Error reading from webcam: {e}")
                    currentState.config(text="Current State: Error reading webcam")
                    attempts += 1
                    if attempts >= max_attempts:
                        print("Max attempts reached. Unable to read from webcam.")
                        currentState.config(text="Current State: Webcam Error - Max Attempts Reached")
                        break 
                    continue  # Skip the rest of the loop and try again
            if ret:
                canny = getCanny(original_frame)
                segment, mask = getSegment(canny)
                colored_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # Convert to a 3 channel image
                overlay = cv.addWeighted(original_frame, 1, colored_mask, 0.3, 0)  # Blend the original frame with the mask

                new_threshold = 50  # Adjust this value as needed
                hough = cv.HoughLinesP(segment, 2, np.pi / 180, new_threshold, np.array([]), minLineLength=100, maxLineGap=50)

                #lines = generateLines(overlay, hough)  # Pass the overlay frame to generateLines
                lines = generateLines(original_frame, hough)  # Pass the original frame to generateLines
                if lines is not None:
                    #processed_frame = showLines(overlay, lines)  # Draw lines on the overlay
                    processed_frame = showLines(original_frame, lines)  # Draw lines on the origianl 
                else:
                    #processed_frame = overlay  # If no lines are detected, use the overlay frame
                    processed_frame = original_frame  # If no lines are detected, use the orginal frame                    

                rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
                rgb_frame = cv.resize(rgb_frame, (256, 256))
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                video_label1.config(image=photo)
                video_label1.image = photo  # Keep a reference
                root.after(0, update_label, Image.fromarray(rgb_frame), video_label1)
