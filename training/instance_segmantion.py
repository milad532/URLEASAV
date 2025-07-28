import carla
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import queue
from datetime import datetime # Import datetime for timestamping folders

# Assuming actors.py is in the same directory and contains walker_blueprints
from actors import walker_blueprints

# Connect to CARLA
client = carla.Client('localhost', 4732)
client.set_timeout(10.0)
town = 'Town02'
world = client.load_world(town)
blueprint_library = world.get_blueprint_library()

# Set up synchronous mode for synchronized sensor data
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# World variables
actor_list = []
walkers = walker_blueprints
walker_spawn_point = [34, 36]

# Create a unique output directory for this run based on current time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = f"carla_run_data/{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)
print(f"Saving data to: {base_output_dir}")

# Spawn a vehicle
vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[40]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
actor_list.append(vehicle)

# Sensor parameters for consistency
IMAGE_X = 800
IMAGE_Y = 600
FOV = 90.0
SENSOR_TRANSFORM = carla.Transform(carla.Location(x=2.0, z=1.8)) # Position on the vehicle

# --- Instance Segmentation Sensor ---
output_folder_instance = os.path.join(base_output_dir, 'instance_segmentation_images')
os.makedirs(output_folder_instance, exist_ok=True)
instance_image_queue = queue.Queue()

def instance_segmentation_callback(image):
    instance_image_queue.put(image)



instance_seg_bp = blueprint_library.find('sensor.camera.instance_segmentation')
instance_seg_bp.set_attribute('image_size_x', str(IMAGE_X))
instance_seg_bp.set_attribute('image_size_y', str(IMAGE_Y))
instance_seg_bp.set_attribute('fov', str(FOV))
instance_segmentation_sensor = world.spawn_actor(instance_seg_bp, SENSOR_TRANSFORM, attach_to=vehicle)
instance_segmentation_sensor.listen(instance_segmentation_callback)
actor_list.append(instance_segmentation_sensor)

# --- Depth Sensor ---
output_folder_depth = os.path.join(base_output_dir, 'depth_images')
os.makedirs(output_folder_depth, exist_ok=True)
depth_image_queue = queue.Queue()

def depth_callback(image):
    depth_image_queue.put(image)

depth_bp = blueprint_library.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', str(IMAGE_X))
depth_bp.set_attribute('image_size_y', str(IMAGE_Y))
depth_bp.set_attribute('fov', str(FOV))
depth_sensor = world.spawn_actor(depth_bp, SENSOR_TRANSFORM, attach_to=vehicle)
depth_sensor.listen(depth_callback)
actor_list.append(depth_sensor)

# --- Helper functions for coordinate transformation ---
def get_matrix(transform):
    """
    Creates a transformation matrix from a carla.Transform object.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return np.matrix(matrix) # Ensure it's a numpy matrix for @ operator

def pixel_to_world(pixel_x, pixel_y, depth_value, camera_transform, K_matrix):
    """
    Converts a pixel from camera image coordinates to world coordinates.
    Args:
        pixel_x (int): X-coordinate of the pixel.
        pixel_y (int): Y-coordinate of the pixel.
        depth_value (float): Depth value at the pixel (in meters).
        camera_transform (carla.Transform): The transform of the camera in the world.
        K_matrix (np.array): Camera intrinsic matrix.
    Returns:
        carla.Location: The 3D world coordinates.
    """
    # Create the point in camera coordinates (homogeneous)
    # Undistort and normalize pixel coordinates
    point_in_camera_homog = np.array([
        (pixel_x - K_matrix[0, 2]) / K_matrix[0, 0],
        (pixel_y - K_matrix[1, 2]) / K_matrix[1, 1],
        1.0
    ])

    # Scale by depth to get actual camera coordinates
    point_in_camera_3d = point_in_camera_homog * depth_value

    # Add a 1 for homogeneous coordinates (x, y, z, 1)
    point_in_camera_homog_full = np.append(point_in_camera_3d, 1.0).reshape(4, 1)

    # Convert to world coordinates using the camera's transform (from local camera to world)
    camera_to_world_matrix = get_matrix(camera_transform)
    
    point_in_world_homog = camera_to_world_matrix @ point_in_camera_homog_full

    return carla.Location(x=float(point_in_world_homog[0]), y=float(point_in_world_homog[1]), z=float(point_in_world_homog[2]))


# Calculate camera intrinsic matrix (K) once
def get_camera_intrinsic_matrix(image_x, image_y, fov):
    f_x = image_x / (2.0 * np.tan(fov * np.pi / 360.0))
    f_y = image_y / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = image_x / 2.0
    cy = image_y / 2.0
    K = np.array([
        [f_x, 0.0, cx],
        [0.0, f_y, cy],
        [0.0, 0.0, 1.0]
    ])
    return K

K_matrix = get_camera_intrinsic_matrix(IMAGE_X, IMAGE_Y, FOV)

# Function to process synchronized sensor data
def process_sensor_data(instance_image, depth_image, world_map, camera_transform, K_matrix):
    # Decode instance segmentation image
    instance_raw_data = np.array(instance_image.raw_data).reshape((IMAGE_Y, IMAGE_X, 4))
    
    instance_ids = (instance_raw_data[:, :, 0].astype(np.uint32) << 16) | \
                   (instance_raw_data[:, :, 1].astype(np.uint32) << 8) | \
                   instance_raw_data[:, :, 2].astype(np.uint32)
    
    semantic_tags = instance_raw_data[:, :, 0]
    # raise ValueError((np.unique(semantic_tags)))
    print(np.unique(semantic_tags))
    # Decode depth image
    depth_raw_data = np.array(depth_image.raw_data).reshape((IMAGE_Y, IMAGE_X, 4))
    # Correct depth calculation: R, G, B are 8-bit.
    # The depth value is (R + G * 256 + B * 256 * 256) / (256*256*256 - 1) * 1000 meters
    depth_values = (depth_raw_data[:, :, 2].astype(np.float32) * (256 * 256) +
                    depth_raw_data[:, :, 1].astype(np.float32) * 256 +
                    depth_raw_data[:, :, 0].astype(np.float32)) / ((2**24) - 1)
    depth_meters = depth_values * 1000.0 # Convert to meters
    
    # Handle potential 0 or very small depths (can cause issues with division)
    # A small epsilon prevents division by zero or very large numbers if depth is near 0
    depth_meters[depth_meters < 0.1] = 0.1 # Minimum depth to avoid issues

    pedestrians_in_lanes = {}
    
    pedestrian_semantic_tag = carla.CityObjectLabel.Pedestrians
    pedestrian_pixels_mask = (semantic_tags == pedestrian_semantic_tag)
    
    unique_pedestrian_instances = np.unique(instance_ids[pedestrian_pixels_mask])

    for instance_id in unique_pedestrian_instances:
        if instance_id == 0: # Instance ID 0 is usually background or unlabelled
            continue
        
        instance_mask = (instance_ids == instance_id)
        combined_mask = pedestrian_pixels_mask & instance_mask

        if not np.any(combined_mask):
            continue

        pixel_y_coords, pixel_x_coords = np.where(combined_mask)
        
        # Strategy: Sample multiple points along the lower part of the pedestrian
        # This makes it more robust to partial occlusions or bad single-point estimations.
        num_samples = 5 # Number of points to sample along the width
        
        # Get the range of x and y coordinates for this pedestrian instance
        min_x, max_x = np.min(pixel_x_coords), np.max(pixel_x_coords)
        min_y, max_y = np.min(pixel_y_coords), np.max(pixel_y_coords)
        
        # Define a region of interest in the lower 20-30% of the pedestrian's height
        sample_y_start = int(max_y - (max_y - min_y) * 0.3)
        sample_y_end = max_y # Go all the way to the bottom
        
        # Ensure sample_y_start is not less than min_y
        if sample_y_start < min_y:
            sample_y_start = min_y
        
        # Select pixels only within this lower region for sampling
        valid_sample_pixels_y, valid_sample_pixels_x = np.where(
            (instance_ids == instance_id) & # Only this instance
            (semantic_tags == pedestrian_semantic_tag) & # Only pedestrians
            (np.arange(IMAGE_Y)[:, None] >= sample_y_start) & # Within the lower Y bound
            (np.arange(IMAGE_Y)[:, None] <= sample_y_end) # Within the upper Y bound
        )

        if len(valid_sample_pixels_x) == 0:
            # Fallback to the single bottom-center if no pixels found in the lower band (e.g., very small pedestrian)
            bottom_pixel_y = np.max(pixel_y_coords)
            x_at_bottom_y = pixel_x_coords[pixel_y_coords == bottom_pixel_y]
            if len(x_at_bottom_y) > 0:
                valid_sample_pixels_x = [int(np.mean(x_at_bottom_y))]
                valid_sample_pixels_y = [bottom_pixel_y]
            else:
                continue # No valid pixels at all for this instance
        
        # Randomly sample points from the lower region (or take evenly spaced if preferred)
        if len(valid_sample_pixels_x) > num_samples:
            indices = np.random.choice(len(valid_sample_pixels_x), num_samples, replace=False)
            sample_x_coords = valid_sample_pixels_x[indices]
            sample_y_coords = valid_sample_pixels_y[indices]
        else:
            sample_x_coords = valid_sample_pixels_x
            sample_y_coords = valid_sample_pixels_y

        found_lane = False
        for px, py in zip(sample_x_coords, sample_y_coords):
            depth_at_pixel = depth_meters[py, px]
            pedestrian_world_location = pixel_to_world(
                px, py, depth_at_pixel, camera_transform, K_matrix
            )

            # Debugging: Draw points in CARLA world for detected pedestrian locations
            # world.debug.draw_point(pedestrian_world_location, size=0.1, color=carla.Color(255,0,0), life_time=0.1)
            
            waypoint = world_map.get_waypoint(pedestrian_world_location, project_to_road=False)
            
            if waypoint:
                lane_id = waypoint.lane_id
                road_id = waypoint.road_id
                lane_type = waypoint.lane_type.name
                
                lane_info = f"Road: {road_id}, Lane: {lane_id}, Type: {lane_type}"
                
                if lane_info not in pedestrians_in_lanes:
                    pedestrians_in_lanes[lane_info] = 0
                pedestrians_in_lanes[lane_info] += 1
                found_lane = True
                break # Count this pedestrian once and move to the next instance
        
        if not found_lane:
            # Optional: Debug when a pedestrian is detected in image but no lane is found
            # print(f"DEBUG: Pedestrian instance {instance_id} detected but no lane found for sampled points.")
            pass # No lane found for this pedestrian after sampling all points

    return pedestrians_in_lanes

# Spawn actors (walkers) - keeping this part the same as it's robust
def spawn_actor(number, list2spawn, list4points, yaw=0):
    point = spawn_points[list4points]
    point.rotation.yaw += yaw
    list_of_points = []
    y_offsets = [(i)/2 for i in range(-number//2, number//2)]
    x_offsets = [(-1)**(i%2) for i in range(-number//2, number//2)]

    for i in range(number):
        list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i], y=point.location.y+y_offsets[i], z=point.location.z), point.rotation))

    for item, point in zip(list2spawn, list_of_points):
        blueprint = blueprint_library.find(item)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        actor = world.try_spawn_actor(blueprint, point)
        if actor:
            actor_list.append(actor)

def spawn_actor_sidewalk(number, list2spawn, list4points, leftright, yaw=0):
    point = spawn_points[list4points]
    point.rotation.yaw += yaw
    point.location.y += (leftright) * 3
    list_of_points = []
    y_offsets = [(i)/2 for i in range(-number//2, number//2)]
    x_offsets = [(-1)**(i%2) for i in range(-number//2, number//2)]

    for i in range(number):
        list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i], y=point.location.y+y_offsets[i], z=point.location.z), point.rotation))

    for item, point in zip(list2spawn, list_of_points):
        blueprint = blueprint_library.find(item)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        actor = world.try_spawn_actor(blueprint, point)
        if actor:
            actor_list.append(actor)

spawn_actor(4, walkers, walker_spawn_point[0])
spawn_actor(5, walkers, walker_spawn_point[1])
spawn_actor_sidewalk(4, walkers, walker_spawn_point[0], -1)
spawn_actor_sidewalk(5, walkers, walker_spawn_point[1], 1)

# Attach a LiDAR sensor (kept for completeness as in your original code)
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('points_per_second', '56000')
lidar_bp.set_attribute('upper_fov', '10')
lidar_bp.set_attribute('lower_fov', '-30')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
actor_list.append(lidar_sensor)

# LiDAR visualization and save
lidar_output_folder = os.path.join(base_output_dir, 'lidar_images')
os.makedirs(lidar_output_folder, exist_ok=True)
lidar_frame_num = 0

def save_lidar_image(lidar_data, frame=0):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=z, s=0.1, cmap='viridis')
    plt.axis('off')
    plt.title(f'LiDAR Frame {frame}')
    plt.savefig(os.path.join(lidar_output_folder, f'lidar_frame_{frame:04d}.png'), bbox_inches='tight')
    plt.close()

    sensor_location = lidar_sensor.get_transform().location
    for px, py, pz in zip(x[::200], y[::200], z[::200]):
        end = carla.Location(x=px + sensor_location.x, y=py + sensor_location.y, z=pz + sensor_location.z)
        world.debug.draw_line(sensor_location, end, thickness=0.02, color=carla.Color(255, 0, 0), life_time=0.1)

def lidar_callback(data):
    global lidar_frame_num
    # save_lidar_image(data, frame=lidar_frame_num) # Uncomment to save LiDAR images
    lidar_frame_num += 1

lidar_sensor.listen(lidar_callback)

# Traffic manager setup
tm = client.get_trafficmanager(8000)

# Run simulation
try:
    print("Simulation running... Press Ctrl+C to stop.")
    # Vehicle will drive forward
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

    world_map = world.get_map()
    frame_count = 0

    while True:
        world.tick()

        try:
            instance_image = instance_image_queue.get(True, 1.0)
            depth_image = depth_image_queue.get(True, 1.0)
        except queue.Empty:
            print("Sensor data not received in time, skipping frame.")
            continue

        if instance_image.frame != depth_image.frame:
            print("Warning: Instance and Depth images are not from the same frame! Skipping.")
            continue
        
        # Save images into the current run's specific folders
        instance_image.save_to_disk(os.path.join(output_folder_instance, f'{instance_image.frame:05d}.png'))
        depth_image.save_to_disk(os.path.join(output_folder_depth, f'{depth_image.frame:05d}.png'))

        camera_transform = instance_segmentation_sensor.get_transform()
        pedestrians_per_lane = process_sensor_data(instance_image, depth_image, world_map, camera_transform, K_matrix)

        if pedestrians_per_lane:
            print(f"\n--- Frame {instance_image.frame} ---")
            for lane_info, count in pedestrians_per_lane.items():
                print(f"Pedestrians in {lane_info}: {count}")
        else:
            print(f"Frame {instance_image.frame}: No pedestrians detected in any specific lane.")

        frame_count += 1
        # if frame_count > 300: # Limit frames for testing (e.g., 15 seconds at 0.05s fixed_delta)
        #     break

finally:
    # Reset to asynchronous mode before destroying actors
    settings.synchronous_mode = False
    world.apply_settings(settings)

    print("\nCleaning up actors...")
    for actor in actor_list:
        if actor and actor.is_alive:
            try:
                actor.destroy()
            except RuntimeError as e:
                print(f"Error destroying actor {actor.id}: {e}")
    print("Cleaned up actors.")