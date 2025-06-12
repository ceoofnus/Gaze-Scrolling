
# take in frame and return eye images and head pose
# to be fed into CNN model to produce gaze coordinates
import dlib

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
    
  

    # SolvePnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, None
    
    # Only use the first three components of rotation_vector for head_pose input
    head_pose = rotation_vector.flatten()[:3]
    
    return left_eye_image, right_eye_image, head_pose