import numpy as np


# Tag center map used for calculating camera positions
#   key = id
#   value = [[x, y, z], [normal_x, normal_y, normal_z]] of the tag center
tag_center_map = {
    0: [[2.23, 0, 1.47], [0, 1, 0]],
    2: [[0, 1.05, 1.41], [1, 0, 0]],
    4: [[0.98, 3.65, 1.32], [0, -1, 0]],
}

# Tag size used for calculating camera positions
tag_size = 0.07  # Unit: [m]

# Minimum axis length for 3d plot visualzation
min_x_length = 5  # Unit: [m]
min_y_length = 5  # Unit: [m]
min_z_length = 3  # Unit: [m]

# Tag size used for visualization only
tag_visualization_size = 0.2  # Unit: [m]
camera_direction_length = 0.5  # Unit: [m]

# Camera configs
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
