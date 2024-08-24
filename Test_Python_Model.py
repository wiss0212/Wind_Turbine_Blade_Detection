import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "C:/Users/daouiaouissem/Desktop/Test_Technique/yolov5/runs/train/mode_yolo5/weights/best.onnx"
session = ort.InferenceSession(model_path)

# Load class names
with open("C:/Users/daouiaouissem/Desktop/Yolo3/labels.name", 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess the input frame
def preprocess(image):
    input_shape = (640, 640)
    image_resized = cv2.resize(image, input_shape)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_expanded = np.expand_dims(image_transposed, axis=0)
    return image_expanded

# Function to postprocess the outputs
def postprocess(image, outputs, score_threshold=0.09):
    boxes = []
    confidences = []
    class_ids = []

    input_height, input_width = image.shape[:2]
    x_factor = input_width / 640
    y_factor = input_height / 640

    detections = outputs[0][0]
    for detection in detections:
        confidence = detection[4]
        if confidence > score_threshold:
            classes_scores = detection[5:]
            class_id = np.argmax(classes_scores)
            class_score = classes_scores[class_id]
            if class_score > score_threshold:
                confidences.append(confidence)
                class_ids.append(class_id)
                
                cx, cy, w, h = detection[:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.1)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []

    return [(class_ids[i], confidences[i], boxes[i]) for i in indices]

# Open video file
video_path = "C:/Users/daouiaouissem/Desktop/Test_Technique/DataPart2/DataPart2/MAH02371.MP4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
output_path = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_image = preprocess(frame)

    # Perform inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})

    # Postprocess the outputs
    detections = postprocess(frame, outputs)

    # Draw bounding boxes
    for class_id, confidence, box in detections:
        left, top, width, height = box
        right = left + width
        bottom = top + height

        label = f'{class_names[class_id]}: {confidence:.2f}'
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Write the frame into the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
