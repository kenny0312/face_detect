# real-time face_reconigtion Using yolov8 and facenet


Pretraining Workflow
Face-alignment: 
1.Face Detection & Alignment 
 Utilize MTCNN to detect facial landmarks and align faces for consistency in input representation.

2.Feature Extraction
 Pass aligned face images through a ResNet-based FaceNet model to extract robust facial embeddings.

3.Loss Functions
 Use a combination of Triplet Loss to maximize inter-class distance and minimize intra-class distance,
And CrossEntropy Loss to ensure correct classification of identities.


object_detection:

1.objects Detection in Videos Using YOLO
 Integrate YOLO (You Only Look Once) to enhance face detection performance in video streams, enabling efficient and scalable analysis.

2. predprocess the image of the obection and using the facenet the get the embedding (features tensor), compare with the face uploaded in my own dataset to identify the information of the object.


