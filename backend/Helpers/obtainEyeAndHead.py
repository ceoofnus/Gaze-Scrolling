
# take in frame and return eye images and head pose
# to be fed into CNN model to produce gaze coordinates
import dlib
import cv2              
import numpy as np


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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

    # Define 3D model points for head pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
  

    # SolvePnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, None
    
    # Only use the first three components of rotation_vector for head_pose input
    head_pose = rotation_vector.flatten()[:3]
    
    return left_eye_image, right_eye_image, head_pose