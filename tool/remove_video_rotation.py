import cv2

# Input and output video paths
input_video_path = "input.mp4"
output_video_path = "output.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check rotation metadata (if available)
rotation_code = None
try:
    rotation_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
except:
    pass

# Determine the new dimensions after rotation
if rotation_code == 90 or rotation_code == 270:
    # Swap width and height for portrait orientation
    new_width, new_height = height, width
else:
    new_width, new_height = width, height

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate the frame based on the rotation code
    if rotation_code == 90:  # Rotated 90 degrees counterclockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif (
        rotation_code == 270
    ):  # Rotated 270 degrees counterclockwise (or 90 degrees clockwise)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_code == 180:  # Upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Write the rotated frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f"Video saved to {output_video_path}")
