# take in CNN output (gaze vector) and convert to onscreen coordinates
import cv2
import numpy as np

def gaze_to_screen(gaze_vector, camera_matrix, dist_coeffs, rvec, tvec):
    gaze_vector = np.array(gaze_vector).reshape(-1, 3)
    print(f"Gaze Vector (Before Projection): {gaze_vector}")
    
    screen_coords, _ = cv2.projectPoints(gaze_vector, rvec, tvec, camera_matrix, dist_coeffs)
    print(f"Screen Coordinates (After Projection): {screen_coords}")

    x_py, y_py = map(float, screen_coords.ravel()[:2])
    return {"x": x_py, "y": y_py}


