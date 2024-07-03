# Gaze-Scrolling
A project I worked on in summer, inspired by Apple's Eye Tracking update on ios devices released May this year. 
Utilising Computer Vision, Deep Learning and image processing, this script allows users to scroll a page with their eye movements!

# How it works
1. The dataset MPIIGaze was chosen due to the variety in lighting conditions and head poses of the subjects. Target variable in this case is the predicted gaze vector of the user, based on eye images and head pose.
2. A Tensorflow CNN model was trained on Colab with the images in MPIIGaze and saved. The Functional (instead of Sequential) API was called due to multimodal nature (input contains both head pose and eye images).
3. Intrinsic camera parameters (camera matrix, rotation + translation vectors, distortion coeff) obtained via pictures taken by laptop webcam. Necessary for transformation from predicted 3D gaze vectors to 2D on-screen coordinates.
4. A pre-trained frontal face detector was imported from dlib, producing eye images and head pose vector.
5. Script assembled in Python 3.8 IDLE + auxilliary functions added 
