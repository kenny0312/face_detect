# face_detect

Pretraining Workflow
1.Face Detection & Alignment 
 Utilize MTCNN to detect facial landmarks and align faces for consistency in input representation.

2.Feature Extraction
 Pass aligned face images through a ResNet-based FaceNet model to extract robust facial embeddings.

3。Loss Functions
 Use a combination of Triplet Loss to maximize inter-class distance and minimize intra-class distance,
And CrossEntropy Loss to ensure correct classification of identities.

Next object:
1.Model Training and Evaluation with Labels
 Successfully train the FaceNet model on labeled face data and evaluate its performance in recognizing identities.

2.Real-time Face Recognition via Webcam
 Implement a pipeline to perform real-time face detection and recognition using a live camera feed.

3.Face Detection in Videos Using YOLO
 Integrate YOLO (You Only Look Once) to enhance face detection performance in video streams, enabling efficient and scalable analysis.
