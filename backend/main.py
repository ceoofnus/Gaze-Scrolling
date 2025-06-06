import pyautogui
import tensorflow as tf

import cv2
import dlib
import numpy as np
import pyautogui

detector = dlib.get_frontal_face_detector()

print(tf.__version__)
# update with path of where the mdoel is stored on your machine
predictor = dlib.shape_predictor(r"C:\Users\zhaox\Downloads\shape_predictor_68_face_landmarks.dat")


dist_coeffs =  np.array([[ 1.16603278e+00, -2.57544176e+01,  5.02202043e-03,  9.03556267e-03,
1.67081879e+02]])
    
camera_matrix = np.array([[1.80304400e+03, 0.00000000e+00, 6.18630102e+02],
[0.00000000e+00, 1.79626830e+03, 3.00968604e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

rvec = np.array([[-0.25410708],
       [ 0.33550616],
       [ 3.10371462]])

tvec = np.array([[ 4.43475453],
       [ 3.47038079],
       [25.13409626]])


# Define 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])


# Define the target shape for eye images
target_shape = (36, 60)  


def get_eye_images_and_head_pose(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, None, None

    
    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Extract and resize eye regions
    left_eye_image = frame[landmarks[37][1]:landmarks[41][1], landmarks[36][0]:landmarks[39][0]]
    right_eye_image = frame[landmarks[43][1]:landmarks[47][1], landmarks[42][0]:landmarks[45][0]]

    left_eye_image = cv2.resize(left_eye_image, (60, 36))
    right_eye_image = cv2.resize(right_eye_image, (60, 36))

    # Ensure images are grayscale
    left_eye_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)
    right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)

    # Add channel dimension for grayscale images
    left_eye_image = np.expand_dims(left_eye_image, axis=-1)
    right_eye_image = np.expand_dims(right_eye_image, axis=-1)

    # Visualize eye images
    cv2.imshow("Left Eye", left_eye_image)
    cv2.imshow("Right Eye", right_eye_image)
    cv2.waitKey(1)
    
    # 2D image points
    image_points = np.array([
        landmarks[30],     # Nose tip
        landmarks[8],      # Chin
        landmarks[36],     # Left eye left corner
        landmarks[45],     # Right eye right corner
        landmarks[48],     # Left Mouth corner
        landmarks[54]      # Right mouth corner
    ], dtype="double")
    
  

    # SolvePnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, None
    
    # Only use the first three components of rotation_vector for head_pose input
    head_pose = rotation_vector.flatten()[:3]
    
    return left_eye_image, right_eye_image, head_pose



def gaze_to_screen(gaze_vector, camera_matrix, dist_coeffs, rvec, tvec):
    gaze_vector = np.array(gaze_vector).reshape(-1, 3)
    print(f"Gaze Vector (Before Projection): {gaze_vector}")
    
    screen_coords, _ = cv2.projectPoints(gaze_vector, rvec, tvec, camera_matrix, dist_coeffs)
    print(f"Screen Coordinates (After Projection): {screen_coords}")

    return screen_coords.ravel()

def scroll_webpage(screen_coords, screen_width, screen_height):
    x, y = screen_coords
    top_threshold = screen_height * 0.2
    bottom_threshold = screen_height * 0.8


    # adjust threshold according to your screen pixel dimensions
    if y >= 538:  
        print("SCROLLING UP")
        pyautogui.scroll(1500)  # Scroll up
    elif y <= 537:
        print("SCROLLING DOWN")
        pyautogui.scroll(-1500)  # Scroll down
    else:
        print("NO SCROLLING")
        pass  # No scrolling

# update with path of where CNN_model is stored on your machine
model = tf.keras.models.load_model(r"C:\Users\zhaox\Downloads\CNN_model")

cap = cv2.VideoCapture(0)

# Get screen dimensions
sw, sh = pyautogui.size()

print(f"Screen Width: {sw} pixels")
print(f"Screen Height: {sh} pixels")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract eye images and head pose from the frame
    left_eye_image, right_eye_image, head_pose = get_eye_images_and_head_pose(frame)

    if left_eye_image is None or right_eye_image is None or head_pose is None:
        continue
    
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

    # Print screen coordinates
    print(f"Screen Coordinates: {screen_coords}")
    
    # Implement scrolling based on screen coordinates
    scroll_webpage(screen_coords, sw, sh)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
