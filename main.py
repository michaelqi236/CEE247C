import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import apriltag
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from configs import *

# Argument parser for video input
parser = argparse.ArgumentParser(
    description="Play video and plot camera's position while Apritag exists."
)

parser.add_argument("video_path", type=str, help="Path to the video file.")
parser.add_argument(
    "--skip_frame",
    type=int,
    default=1,
    help="frames to skip between two plots. (default: 1)",
)
args = parser.parse_args()


def rotate_frame(frame, rotation_code):
    # Rotate the frame if necessary
    if rotation_code == 90:  # Video is rotated 90 degrees counterclockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif (
        rotation_code == 270
    ):  # Video is rotated 270 degrees counterclockwise (or 90 degrees clockwise)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_code == 180:  # Video is upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# Initialize empty figures
def create_figures(cap, rotation_code):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Convert second subplot to 3D
    ax2.axis("off")
    ax2 = fig.add_subplot(122, projection="3d")

    # Set 3D plot limits
    coordinates = [tag_center[0] for tag_center in tag_center_map.values()]
    x_values, y_values, z_values = zip(*coordinates)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    x_max = x_mid + max(x_max - x_mid, min_x_length / 2)
    x_min = x_mid - max(x_mid - x_min, min_x_length / 2)
    y_max = y_mid + max(y_max - y_mid, min_y_length / 2)
    y_min = y_mid - max(y_mid - y_min, min_y_length / 2)
    z_max = z_mid + max(z_max - z_mid, min_z_length / 2)
    z_min = z_mid - max(z_mid - z_min, min_z_length / 2)

    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([z_min, z_max])
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # Initialize an empty image plot
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        exit()

    # Initialize April tag plot
    tag_corners_viz_map = compute_tag_corners(tag_center_map, tag_visualization_size)
    for tag_id, tag_corners in tag_corners_viz_map.items():
        # Define polygon and add to plot
        tag_face = Poly3DCollection(
            [tag_corners], color="cyan", alpha=0.5, edgecolor="black"
        )
        ax2.add_collection3d(tag_face)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert first frame to RGB
    frame = rotate_frame(frame, rotation_code)

    frame_display = ax1.imshow(frame)
    ax1.axis("off")  # Hide axis for video display

    return fig, frame_display, ax1, ax2


def update_3d_plot(ax2, tag_id, position, direction, camera_trajectory, lines):
    # Clear previous plot
    for line in lines:
        line.remove()
    lines.clear()

    # Draw position
    if position is not None:
        tag_center = tag_center_map[tag_id][0]
        (line,) = ax2.plot(
            [tag_center[0], position[0]],
            [tag_center[1], position[1]],
            [tag_center[2], position[2]],
            "b",
            lw=2,
        )
        lines.append(line)

    # Draw direction
    if position is not None and direction is not None:
        (line,) = ax2.plot(
            [position[0], position[0] + camera_direction_length * direction[0]],
            [position[1], position[1] + camera_direction_length * direction[1]],
            [position[2], position[2] + camera_direction_length * direction[2]],
            "r",
            lw=2,
        )
        lines.append(line)

    # Draw camera trajectory
    if camera_trajectory.shape[0] > 1:
        (line,) = ax2.plot(
            camera_trajectory[:, 0],
            camera_trajectory[:, 1],
            camera_trajectory[:, 2],
            "gray",
            lw=1,
        )
        lines.append(line)

    return lines


def quaternion_from_vector(v):
    # Normalize the target vector to make sure it's a unit vector
    v = v / np.linalg.norm(v)

    # Compute the axis of rotation: cross product between [1, 0, 0] and the vector
    axis = np.cross([1, 0, 0], v)

    # If the vectors are parallel (i.e., the cross product is zero), no rotation is needed
    if np.linalg.norm(axis) == 0:
        # If they are already aligned, no rotation is needed, return identity quaternion
        return R.from_quat([0, 0, 0, 1]).as_quat()

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Compute the angle between the vectors using dot product
    angle = np.arccos(np.dot([1, 0, 0], v))

    # Create the quaternion from the axis and angle
    return R.from_rotvec(axis * angle).as_quat()


def compute_tag_corners(tag_center_map, tag_size):
    """
    Computes the 3D corner coordinates of square AprilTags using their center and normal.
    """
    tag_corners_map = {}

    for tag_id, (tag_center, tag_normal) in tag_center_map.items():
        tag_center = np.array(tag_center)
        tag_normal = np.array(tag_normal)
        tag_normal = tag_normal / np.linalg.norm(tag_normal)  # Normalize normal vector

        # Compute the four corner points relative to the center
        half_size = tag_size / 2
        local_corners = np.array(
            [
                [0, -half_size, half_size],  # Top-left
                [0, half_size, half_size],  # Top-right
                [0, half_size, -half_size],  # Bottom-right
                [0, -half_size, -half_size],  # Bottom-left
            ]
        )

        q = quaternion_from_vector(tag_normal)
        rotation = R.from_quat(q)
        rotated_corners = rotation.apply(local_corners)

        # Shift the rotated corners to the tag's center position
        world_corners = rotated_corners + tag_center

        # Store in dictionary
        tag_corners_map[tag_id] = world_corners

    return tag_corners_map


def get_camera_pose_from_apriltag(results, frame, tag_corners_map):
    if len(results) > 0:
        # Get the first detected tag
        r = results[0]
        tag_id = r.tag_id
        corners = r.corners

        # Draw the detected tag boundary
        for i in range(4):
            cv2.line(
                frame,
                tuple(corners[i].astype(int)),
                tuple(corners[(i + 1) % 4].astype(int)),
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"tag_id: {tag_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if tag_id in tag_corners_map:
            # Estimate the pose of the tag
            ret, rvec, tvec = cv2.solvePnP(
                tag_corners_map[tag_id], corners, camera_matrix, dist_coeffs
            )

            # Compute the camera position and direction in the tag's coordinate frame
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                R_inv = np.linalg.inv(R)
                camera_position = (R_inv @ -tvec).flatten()
                camera_direction = R.T[:, 2]

                cv2.putText(
                    frame,
                    f"Camera position: {camera_position}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Camera direction: {camera_direction}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                return camera_position, camera_direction, tag_id
    return None, None, None


def main():
    # Open the video file
    print(f"Video file path: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    rotation_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    tag_corners_map = compute_tag_corners(tag_center_map, tag_size)

    # Create figure with two subplots
    fig, frame_display, _, ax2 = create_figures(cap, rotation_code)
    lines = []
    camera_trajectory = np.empty((0, 3))

    # Main loop to update both plots
    while cap.isOpened():
        # Check if the figure is closed
        if not plt.fignum_exists(fig.number):
            break

        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detect AprilTags in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        camera_position, camera_direction, tag_id = get_camera_pose_from_apriltag(
            results, frame, tag_corners_map
        )

        if camera_position is not None:
            camera_trajectory = np.vstack((camera_trajectory, camera_position))

        # Update video frame
        frame = rotate_frame(frame, rotation_code)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.set_data(frame)

        # Update 3D plot
        lines = update_3d_plot(
            ax2, tag_id, camera_position, camera_direction, camera_trajectory, lines
        )

        plt.draw()  # Force update of figure
        plt.pause(0.01)  # Small delay for smooth updating

        # Skip the next `skip_frame` frames
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get current frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + args.skip_frame)  # Skip frames

    plt.pause(1e6)
    cap.release()
    plt.close()


if __name__ == "__main__":
    main()
