# Gaze-Scrolling
A project I worked on in summer, inspired by Apple's Eye Tracking update on ios devices released May this year. 
Utilising Computer Vision, Deep Learning and image processing, this script allows users to scroll a page with their eye movements!

# How it works
1. The dataset MPIIGaze was chosen due to the variety in lighting conditions and head poses of the subjects. Target variable in this case is the predicted gaze vector of the user, based on eye images and head pose.
2. A Tensorflow CNN model was trained on Colab with the images in MPIIGaze and saved. The Functional (instead of Sequential) API was called due to multimodal nature (input contains both head pose and eye images).
3. Intrinsic camera parameters (camera matrix, rotation + translation vectors, distortion coeff) obtained via pictures taken by laptop webcam. Necessary for transformation from predicted 3D gaze vectors to 2D on-screen coordinates.
4. A pre-trained frontal face detector was imported from dlib, producing eye images and head pose vector.
5. Script assembled in Python 3.8 IDLE + auxilliary functions added

# Usage instructions
1. Download the Main_script py file. Note that the scipt must be run LOCALLY for it to have access to your webcam.
2. Download and unzip my trained CNN_model.
3. Downlaod the dlib facial landmark detector from here: https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/blob/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat
4. Update paths of CNN_model and dlib shape_predictor in the main script.
5. Note that scrolling effects may not be optimal as the script is based off my own laptop's intrinsic webcam parameters. I'm still working on trying to obtain users' parameters.
6. Execute the script and have fun scrolling with your eyes!
