# DeepBhvTracking
A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning.

A strategy to track an animal’s movement by combining a deep learning technique, the You Only Look Once (YOLO) algorithm, with a background subtraction algorithm - DeepBhvTracking. In our method, we first train the detector using manually labeled images and a pre-trained deep-learning neural network combined with YOLO, then generate bounding boxes of the targets using the trained detector, and last track the center of the targets by calculating their centroid in the bounding box using background subtraction. Using DeepBhvTracking, movement of animals can be tracked accurately in complex environments and can be used in different behavior paradigms and for different animal models.

A demo detector trained for black mice could be downloaded in the latest releases.
