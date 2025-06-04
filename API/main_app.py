from fastapi import FastAPI
import numpy as np
import cv2
import tensorflow as tf
#import pyautogui
import os
from Helpers import gaze_to_screen
#from Helpers import scroll_webpage
from Helpers import get_eye_images_and_head_pose


# set of intrinsic camera parameters obtained via camera callibration 
# obtained during model training process

dist_coeffs =  np.array([[ 1.16603278e+00, -2.57544176e+01,  5.02202043e-03,  9.03556267e-03,
1.67081879e+02]])

camera_matrix = np.array([[1.80304400e+03, 0.00000000e+00, 6.18630102e+02],
[0.00000000e+00, 1.79626830e+03, 3.00968604e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

rotational_vec = np.array([[-0.25410708],
       [ 0.33550616],
       [ 3.10371462]])

translational_vec = np.array([[ 4.43475453],
       [ 3.47038079],
       [25.13409626]])




# where the model is stored in the container
model = tf.keras.models.load_model(r"CNN_model")


# actl this shld be left to React frontend to capture
# instead of using webcam, frontend sends POST request to API for processing
#cap = cv2.VideoCapture(0)

# Get screen dimensions
sw = os.environ.get('SCREEN_WIDTH') 
sh = os.environ.get('SCREEN_HEIGHT')    
#sw, sh = pyautogui.size()

app = FastAPI()
# the only necessary endpoint of the API
# takes in captured frame and returns on screen gaze coordinates
@app.post("/")  
async def onscreen_coord(frames):
    try:
        ret, frame = cap.read()    # refactor THIS part to accept POST from frontend    
    except Exception as e:
        return {f"error: I wasn't able to capture a frame from the webcam"}  
    
    # Extract eye images and head pose from the frame
    left_eye_image, right_eye_image, head_pose = get_eye_images_and_head_pose(frame)

    if left_eye_image is None or right_eye_image is None or head_pose is None:
        return {"error": "Could not capture eye images or head pose"}
    
    # Preprocess the images
    left_eye_image = left_eye_image / 255.0
    right_eye_image = right_eye_image / 255.0

    # Expand dimensions to match model input shape
    left_eye_image = np.expand_dims(left_eye_image, axis=0)
    right_eye_image = np.expand_dims(right_eye_image, axis=0)
    head_pose = np.expand_dims(head_pose, axis=0)
 
   # Predict gaze vector
   # rememeber: my trained CNN takes in images and returns gaze vector
    predicted_gaze = model.predict([left_eye_image, head_pose])

    # Print predicted gaze vector
    print(f"Predicted Gaze: {predicted_gaze}")
    
    # Convert predicted gaze vector to screen coordinates
    screen_coords = gaze_to_screen(predicted_gaze, camera_matrix, dist_coeffs, rvec, tvec)

    return screen_coords

@app.get("/testing")
async def test():
    return "API running!"



