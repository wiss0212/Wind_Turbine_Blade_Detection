import cv2
import numpy as np

# Load the video
video_path = "C:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\DataPart2\\DataPart2\\MAH02371.MP4"

cap = cv2.VideoCapture(video_path)

# Pre-process Each Frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Detect Contours
def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Identify the Blades
def filter_blades(contours):
    blades = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Assuming blades are roughly quadrilateral
            blades.append(contour)
    return blades

# Calculate the angle of a point relative to the hub
def calculate_angle(centroid, hub_position):
    return np.arctan2(centroid[1] - hub_position[1], centroid[0] - hub_position[0])

# Assign number  to Blades
def assign_blade_ids(blades, hub_position, last_angles):
    blade_ids = []
    current_angles = []

    for blade in blades:
        M = cv2.moments(blade)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = np.array([cx, cy])

        angle = calculate_angle(centroid, hub_position)
        current_angles.append((blade, angle))

    current_angles.sort(key=lambda x: x[1])

    for i, (blade, angle) in enumerate(current_angles):
        if len(last_angles) >= 3:
            previous_angle = last_angles[i % 3][1]
            if abs(angle - previous_angle) < np.pi / 4:
                blade_ids.append((blade, i % 3 + 1))
            else:
                blade_ids.append((blade, i % 3 + 1))
        else:
            blade_ids.append((blade, i % 3 + 1))

    return blade_ids, current_angles

# Visualize the Detected Blades
def draw_blades(frame, blade_ids):
    for blade, blade_id in blade_ids:
        cv2.drawContours(frame, [blade], -1, (0, 255, 0), 3)
        M = cv2.moments(blade)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(frame, f'Blade {blade_id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

previous_blades = {}
hub_position = (int(cap.get(3) / 2), int(cap.get(4) / 2))  
last_angles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess_frame(frame)
    contours = detect_contours(edges)
    blades = filter_blades(contours)
    blade_ids, last_angles = assign_blade_ids(blades, hub_position, last_angles)
    output_frame = draw_blades(frame, blade_ids)

    # Write the frame to the output video
    out.write(output_frame)

    cv2.imshow('Wind Turbine Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
