#imports packages
from flask import *
import sqlite3
import bcrypt
import time
from flask_cors import CORS
import cv2
import numpy as np
from flask import Response
from threading import Thread

# Attempt to import and initialize MotorKit only on supported platforms
try:
    from adafruit_motorkit import MotorKit
    kit = MotorKit(0x40)
except (NotImplementedError, ImportError):
    print("Running on an unsupported platform. Motor functionality will be mocked.")

    # Define a mock MotorKit for development purposes
    class MockMotorKit:
        def __init__(self, address=0x40):
            print(f"Initialized MockMotorKit at address {hex(address)}")

        @property
        def motor1(self):
            return self.MockMotor()

        @property
        def motor2(self):
            return self.MockMotor()

        class MockMotor:
            def __init__(self):
                self._throttle = 0

            @property
            def throttle(self):
                return self._throttle

            @throttle.setter
            def throttle(self, value):
                self._throttle = value
                print(f"Mock motor throttle set to {value}")

    # Use the mock class instead of the real MotorKit
    kit = MockMotorKit()

 #creates flask app
app = Flask(__name__)
CORS(app)

#used for session
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Global variable for control
is_running = False

#gets database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

camera = cv2.VideoCapture(0)

def getCanny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
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
    cv2.fillPoly(mask, polygons, 255)
    segment = cv2.bitwise_and(frame, mask)
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

def calculate_centerline(left_line, right_line):
    """
    Calculate the centerline based on the left and right lane lines.
    Each line is represented by two points (x1, y1) and (x2, y2).
    """
    if left_line is None or right_line is None:
        return None

    # Calculate midpoints of the left and right lines
    left_midpoint = [(left_line[0] + left_line[2]) / 2, (left_line[1] + left_line[3]) / 2]
    right_midpoint = [(right_line[0] + right_line[2]) / 2, (right_line[1] + right_line[3]) / 2]

    # Calculate the midpoint of the midpoints to find the centerline
    center_x = (left_midpoint[0] + right_midpoint[0]) / 2
    center_y = (left_midpoint[1] + right_midpoint[1]) / 2

    return (center_x, center_y)

def adjust_robot_direction(center_line):
    """
    Adjust the robot's direction based on the centerline position.
    """
    if center_line is None:
        print("Centerline not detected, stopping the robot.")
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        return
    frame_center_x = camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2  # Assuming 'camera' is your VideoCapture object
    # Determine the direction to move based on the centerline's x position relative to the frame's center
    if abs(center_line[0] - frame_center_x) < 20:  # Centerline is close to center, move forward
        print("Moving forward")
        kit.motor1.throttle = 0.732
        kit.motor2.throttle = 0.7
    elif center_line[0] < frame_center_x:  # Centerline is to the left, turn left
        print("Turning left")
        kit.motor1.throttle = 0.72
        kit.motor2.throttle = -0.75
    else:  # Centerline is to the right, turn right
        print("Turning right")
        kit.motor1.throttle = -0.72
        kit.motor2.throttle = 0.72

    # Adjust the duration of the turn/movement if necessary
    time.sleep(0.3)  # Example sleep duration to prevent continuous movement

def lane_following_task():
    global is_running, kit
    is_running = True
    while is_running:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break
        canny = getCanny(frame)
        segment, mask = getSegment(canny)
        new_threshold = 50  # Adjust this value as needed
        hough = cv2.HoughLinesP(segment, 2, np.pi / 180, new_threshold, np.array([]), minLineLength=100, maxLineGap=50)
        lines = generateLines(frame, hough)        
        if lines is not None:
            # Assuming generateLines returns [[x1, y1, x2, y2], [x1, y1, x2, y2]] for left and right lines
            left_line, right_line = lines
            # Here you would calculate the centerline and make a decision on how to adjust the robot's direction.
            # For simplicity, let's assume we just call a function adjust_robot_direction(center_line)
            center_line = calculate_centerline(left_line, right_line)
            adjust_robot_direction(center_line)
        time.sleep(0.1)  # Adjust based on your needs

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#navigates to homepage
@app.route('/')
def index():  
    if 'username' in session:
        return render_template('index.html',firstname=session['username'])
    return render_template('login.html')

#navigates to login page
@app.route('/login', methods = ['GET', 'POST'])
def login():
    #checks user credentials
    if request.method == 'POST':
      conn = get_db_connection()
      user = conn.execute('SELECT * from user where username = ? ',
                          (str(request.form['username']),)).fetchone()
      conn.close()
      #checks password
      if bcrypt.checkpw(request.form["password"].encode("utf-8"),str(user["password"]).encode("utf-8")):
          session['username'] = user["first_name"]
          return redirect(url_for('index'))    
      else:
         print("User/ Password Error")

    return render_template('login.html')

#creates registration page
@app.route('/registration',methods=['GET', 'POST'])
def registration():
    #saves user information
    if request.method == 'POST':
        firstname=request.form["fname"]
        lastname=request.form["lname"]
        username=request.form["username"]
        #encrypts password
        salt = bcrypt.gensalt()
        password = (bcrypt.hashpw(request.form["password"].encode("utf-8"),salt).decode(encoding= "utf-8"))
        conn = get_db_connection()
        user = conn.execute('SELECT * from user where username = ? ',
                          (str(request.form['username']),)).fetchone()
        #checks if user is in the database
        if user is None:
             conn.execute('INSERT INTO user (username,password,first_name,last_name) VALUES (?, ?,?,?)',                          
                         (username,password, firstname, lastname ))
        else:
            print(f"User {username} already exist!")
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    else:
     return render_template('registration.html')

#logs out and redirects to login page
@app.route('/logout')
def logout():
    #removes the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('login'))

lane_following_thread = None
#Start the robor to follow the centerline of a lane based on lane detection    
@app.route('/start', methods=['GET', 'POST'])
def start():
    global lane_following_thread
#    if lane_following_thread is None or not lane_following_thread.is_alive():    
    if lane_following_thread is None:
        lane_following_thread = Thread(target=lane_following_task)
        lane_following_thread.start()
    return jsonify("start")

 #Move the robor right    
@app.route('/right', methods = ['GET', 'POST'])
def right():
  #moves robot right
  kit.motor1.throttle = 0.72 * 1
  kit.motor2.throttle = (0.72 + 0.069) * 1
  #runs both motors for 0.3 seconds
  time.sleep(0.3)
  return jsonify("right")

 #Move the robot forward
@app.route('/forward', methods = ['GET', 'POST'])
def forward():
  #moves robot forward
  kit.motor1.throttle = 0.775
  kit.motor2.throttle = (0.775 - 0.15) * -1
  #runs both motors for 0.3 seconds
  time.sleep(0.3)
  return jsonify("forward")

@app.route('/backward', methods = ['GET', 'POST'])
#Move the robot backward
def backward():
  #moves robot backwards
  kit.motor1.throttle = 0.775 * -1
  kit.motor2.throttle = 0.775 - 0.1
  #runs both motors for 0.3 seconds
  time.sleep(0.3)
  return jsonify("backward")

@app.route('/left', methods = ['GET', 'POST'])
#Move the robot left
def left():
  #moves robot left
  kit.motor1.throttle = 0.72 * -1
  kit.motor2.throttle = (0.72 + 0.069) * -1
  #runs both motors for 0.3 seconds
  time.sleep(0.3)
  return jsonify("left")

#Stop the robot
@app.route('/stop', methods=['GET', 'POST'])
def stop():
    global is_running, lane_following_thread
    is_running = False
    if lane_following_thread is not None:
        lane_following_thread.join()
    kit.motor1.throttle = 0
    kit.motor2.throttle = 0
    return jsonify("stop")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Allows app to run
if __name__ == '__main__':
    app.run(host='192.168.1.14', port=5000) #Try this one first; if not working,try the next line
    #app.run(debug=True, port=5000)
