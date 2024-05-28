import io
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk   
from datetime import datetime
import logging
import pytz
import threading


def thing(image):
    def contours_2(image, og, extra_pix=0):
        # find the contours on the image
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours((contours, hierarchy))
        # sort the list of contours by the contour area
        new_lst = list(contours)
        new_lst.sort(key=cv.contourArea)
        # if there are at least 2 contours that have been detected
        if len(new_lst) > 1:
            # get the 2 largest contours
            c1 = new_lst[-1]
            c2 = new_lst[-2]
            # fit polylines to each contour
            outline1 = cv.approxPolyDP(c1, 4, True)
            cv.drawContours(image, [outline1], -1, (0, 255, 255), 15)
            outline2 = cv.approxPolyDP(c2, 4, True)
            cv.drawContours(image, [outline2], -1, (0, 255, 255), 15)
            # draw a midline by going through the polyline and averaging each x and y coordinate
            # append this averaged coordinate to a list and turn that list into a numpy array
            midline = []

            for pt1, pt2 in zip(outline1[:int(len(outline1) / 1.8)], outline2[:int(len(outline2) / 1.8)]):
                mid_x = int((pt1[0][0] + pt2[0][0]) / 2) + extra_pix
                mid_y = int((pt1[0][1] + pt2[0][1]) / 2)
                midline.append([[mid_x, mid_y]])
            midline = np.array(midline, dtype=np.int32)
            # draw a polyline from the numpy array onto the frame
            cv.polylines(og, [midline], False, (0, 255, 0), 15)
            return midline


    def colorr(image, lower, upper):
        mask = cv.inRange(image, lower, upper)
        masked = cv.bitwise_and(image, image, mask=mask)
        return masked

    def filtering(image):
        mask2 = colorr(image, (50, 50, 60), (250, 250, 250))
        image = image - mask2
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gauss = cv.GaussianBlur(gray, (15, 15), 0)
        gauss = cv.medianBlur(gauss, 15)
        canny2 = cv.Canny(gauss, 30, 60)

        return canny2

    # un-warps an image given a set of vertices
    def unwarped(img, mask_vert, screen_vert):
        matrix2 = cv.getPerspectiveTransform(screen_vert, mask_vert)
        result = cv.warpPerspective(img, matrix2, (img.shape[1], img.shape[0]))
        return result

    # warps an image given a set a vertices
    def warping(image, mask_vert, screen_vert):
        matrixy = cv.getPerspectiveTransform(mask_vert, screen_vert)
        result = cv.warpPerspective(image, matrixy, (image.shape[1], image.shape[0]))
        return result

    height = image.shape[0]  # 1080
    global width
    width = image.shape[1]  # 1920
    p1 = [round(width * .1), round(height * 1)]
    p2 = [round(width * .22), round(height * .28)]
    p3 = [round(width * .79), round(height * .28)]
    p4 = [round(width * .90), round(height * 1)]
    # create a trapezoidal mask around the road
    mask_vertices = np.int32([p1, p2, p3, p4])
    # cv.polylines(image, [mask_vertices], True, (0,0,0), 5)
    screen_verts = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
    # warp the frame to fit this trapezoidal mask to get a bird's-eye view of the road
    warped_image = warping(image, np.float32(mask_vertices), screen_verts)
    filtered = filtering(warped_image)
    crop_l = filtered[0:height, 0:width // 2]

    crop_r = filtered[0:height, width // 2:width]

    leftc = contours_2(crop_l, warped_image)
    rightc = contours_2(crop_r, warped_image, width // 2)
    middle = []
    scalell = []
    maxx = width // 2
    maxy = 0
    if leftc is not None and rightc is not None:
        for x in range(len(leftc)):
            try:
                scalel = int((leftc[x][0][0] + rightc[x][0][0]) / 2) / int((leftc[x][0][0]))
                middle.append(
                    [int((leftc[x][0][0] + rightc[x][0][0]) / 2), int((leftc[x][0][1] + rightc[x][0][1]) / 2)])
                scalell.append(scalel)
            except:
                if scalell != []:
                    scaless = sum(scalell) / len(scalell)
                    middle.append([int((leftc[x][0][0]) * scaless), int((leftc[x][0][1]) * scaless)])
                else:
                    break


        for point in middle:
            if point[1] > maxy:
                maxx = point[0]
        middle = np.array(middle, dtype=np.int32)
        cv.polylines(warped_image, [middle], False, (0, 255, 255), 15)

        unwarped = unwarped(warped_image, np.float32(mask_vertices), screen_verts)

        # add the unwarped image and the orginal image ontop of each other
        finished = cv.addWeighted(image, 0.5, unwarped, 0.5, 0.0)

        return finished

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


def detect_vertical_lines(frame):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect edges using the Canny edge detector
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines using the probabilistic Hough line transform
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:  # Check if any lines are detected
        for line in lines:  # Iterate through each detected line
            for x1, y1, x2, y2 in line:  # Extract the coordinates of the line endpoints
                # Calculate the angle of the line in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Check if the line is approximately vertical (angle close to 90 or -90 degrees)
                if abs(angle) > 80 and abs(angle) < 100:
                    # Draw a rectangle around the detected vertical line
                    cv.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 255, 0), 2)
                    # Put a label "Obstacle" near the detected line
                    cv.putText(frame, 'Obstacle', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame  # Return the frame with detected lines highlighted
cascade_src = 'obstacle.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
def detect_obstacle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    obstacle = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in obstacle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame

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
                processed_frame = thing(processed_frame)
                processed_frame = detect_vertical_lines(processed_frame)
                processed_frame = detect_obstacle(processed_frame)

                rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
                rgb_frame = cv.resize(rgb_frame, (256, 256))
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                video_label1.config(image=photo)
                video_label1.image = photo  # Keep a reference
                root.after(0, update_label, Image.fromarray(rgb_frame), video_label1)
