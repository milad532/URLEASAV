import carla
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import queue
from datetime import datetime
from actors import walker_blueprints # Assuming this correctly imports walker blueprints

# CONNECTIVITY AND SETUP (NO CHANGE)
client = carla.Client('localhost', 4732)
client.set_timeout(10.0)
town = 'Town02'
world = client.load_world(town)
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

actor_list = []
walkers = walker_blueprints
walker_spawn_point = [34, 36]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = f"carla_run_data/{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)
print(f"Saving data to: {base_output_dir}")

vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[40]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
actor_list.append(vehicle)

IMAGE_X = 800
IMAGE_Y = 600
FOV = 90.0
SENSOR_TRANSFORM = carla.Transform(carla.Location(x=2.0, z=1.8))

# --- SEMANTIC Segmentation Sensor (Changed sensor type and queue/callback names) ---
output_folder_semantic = os.path.join(base_output_dir, 'semantic_segmentation_images')
os.makedirs(output_folder_semantic, exist_ok=True)
semantic_image_queue = queue.Queue() # Changed queue name

def semantic_segmentation_callback(image): # Changed callback name
    semantic_image_queue.put(image)

semantic_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation') # Changed sensor type
semantic_seg_bp.set_attribute('image_size_x', str(IMAGE_X))
semantic_seg_bp.set_attribute('image_size_y', str(IMAGE_Y))
semantic_seg_bp.set_attribute('fov', str(FOV))
semantic_segmentation_sensor = world.spawn_actor(semantic_seg_bp, SENSOR_TRANSFORM, attach_to=vehicle)
semantic_segmentation_sensor.listen(semantic_segmentation_callback) # Changed listener
actor_list.append(semantic_segmentation_sensor)

# --- Depth Sensor --- (No change needed)
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

# --- Helper functions for coordinate transformation --- (NO CHANGE)
def get_matrix(transform):
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
    return np.matrix(matrix)

def pixel_to_world(pixel_x, pixel_y, depth_value, camera_transform, K_matrix):
    point_in_camera_homog = np.array([
        (pixel_x - K_matrix[0, 2]) / K_matrix[0, 0],
        (pixel_y - K_matrix[1, 2]) / K_matrix[1, 1],
        1.0
    ])
    point_in_camera_3d = point_in_camera_homog * depth_value
    point_in_camera_homog_full = np.append(point_in_camera_3d, 1.0).reshape(4, 1)
    camera_to_world_matrix = get_matrix(camera_transform)
    point_in_world_homog = camera_to_world_matrix @ point_in_camera_homog_full
    return carla.Location(x=float(point_in_world_homog[0]), y=float(point_in_world_homog[1]), z=float(point_in_world_homog[2]))

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

# Function to process synchronized sensor data using SEMANTIC segmentation
def process_semantic_data(semantic_image, depth_image, world_map, camera_transform, K_matrix):
    # Decode semantic segmentation image: Extract from Alpha channel
    semantic_raw_data = np.array(semantic_image.raw_data).reshape((IMAGE_Y, IMAGE_X, 4))
    semantic_tags = semantic_raw_data[:, :, 3] # Extracting Alpha channel

    print(f"Unique semantic tags in this frame (from Alpha channel): {np.unique(semantic_tags)}")

    # Decode depth image (same as before)
    depth_raw_data = np.array(depth_image.raw_data).reshape((IMAGE_Y, IMAGE_X, 4))
    depth_values = (depth_raw_data[:, :, 2].astype(np.float32) * (256 * 256) +
                    depth_raw_data[:, :, 1].astype(np.float32) * 256 +
                    depth_raw_data[:, :, 0].astype(np.float32)) / ((2**24) - 1)
    depth_meters = depth_values * 1000.0
    depth_meters[depth_meters < 0.1] = 0.1

    pedestrians_in_lanes = {}
    
    pedestrian_semantic_tag = int(carla.CityObjectLabel.Pedestrians)
    
    # Create a binary mask for pedestrian pixels
    pedestrian_pixels_mask = (semantic_tags == pedestrian_semantic_tag)

    if not np.any(pedestrian_pixels_mask):
        return {} # No pedestrians detected in the entire image

    # --- SIMPLIFIED PEDESTRIAN DETECTION WITHOUT SKIMAGE ---
    # This will treat ALL pedestrian pixels as one big "blob" and just try to find a single
    # representative point. This will *not* count individual pedestrians.
    # It will only tell you if *any* pedestrian pixels were detected and
    # attempt to map one point to a lane.

    # Find all pedestrian pixel coordinates
    pedestrian_y_coords, pedestrian_x_coords = np.where(pedestrian_pixels_mask)

    # If no pedestrian pixels, return empty
    if len(pedestrian_x_coords) == 0:
        return {}
    
    # For a simple check, just take the 'bottom-center' pixel of the *entire* pedestrian mask
    # This will only count one "pedestrian presence" per frame, not individual pedestrians.
    # It's primarily for testing if the Alpha channel contains the tag correctly.
    
    bottom_pixel_y = np.max(pedestrian_y_coords)
    x_at_bottom_y = pedestrian_x_coords[pedestrian_y_coords == bottom_pixel_y]
    if len(x_at_bottom_y) == 0:
        return {} # Should not happen if pedestrian_pixels_mask was true

    bottom_center_pixel_x = int(np.mean(x_at_bottom_y))

    # Ensure pixel coordinates are within bounds for depth lookup
    px = np.clip(bottom_center_pixel_x, 0, IMAGE_X - 1)
    py = np.clip(bottom_pixel_y, 0, IMAGE_Y - 1)

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
        
        # With semantic segmentation and no instance ID, we can only count "presence" in a lane,
        # not distinct pedestrians. We'll simply note that pedestrians are present in this lane.
        # If you truly want a count, you'd need the skimage part.
        pedestrians_in_lanes[lane_info] = pedestrians_in_lanes.get(lane_info, 0) + 1 # This will effectively be 1 if found
    # else:
    #     print(f"DEBUG: Pedestrian pixels detected but no lane found for representative point.")

    return pedestrians_in_lanes

# ACTOR SPAWNING (NO CHANGE)
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

# LiDAR sensor (NO CHANGE)
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
    # save_lidar_image(data, frame=lidar_frame_num)
    lidar_frame_num += 1

lidar_sensor.listen(lidar_callback)

tm = client.get_trafficmanager(8000)

# RUN SIMULATION (MODIFIED SENSOR CALLS)
try:
    print("Simulation running... Press Ctrl+C to stop.")
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

    world_map = world.get_map()
    frame_count = 0

    while True:
        world.tick()

        try:
            semantic_image = semantic_image_queue.get(True, 1.0) # Get semantic image
            depth_image = depth_image_queue.get(True, 1.0)
        except queue.Empty:
            print("Sensor data not received in time, skipping frame.")
            continue

        if semantic_image.frame != depth_image.frame:
            print("Warning: Semantic and Depth images are not from the same frame! Skipping.")
            continue
        
        semantic_image.save_to_disk(os.path.join(output_folder_semantic, f'{semantic_image.frame:05d}.png'))
        depth_image.save_to_disk(os.path.join(output_folder_depth, f'{depth_image.frame:05d}.png'))

        camera_transform = semantic_segmentation_sensor.get_transform()
        # Call the new processing function
        pedestrians_per_lane = process_semantic_data(semantic_image, depth_image, world_map, camera_transform, K_matrix)

        if pedestrians_per_lane:
            print(f"\n--- Frame {semantic_image.frame} ---")
            for lane_info, count in pedestrians_in_lanes.items():
                print(f"Pedestrians present in {lane_info}: {count}") # Changed wording
        else:
            print(f"Frame {semantic_image.frame}: No pedestrian presence detected in any specific lane.")

        frame_count += 1
        # if frame_count > 300:
        #     break

finally:
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