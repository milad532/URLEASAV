import numpy as np
import time
import math
import cv2
import os
import carla
import random
client = carla.Client('localhost', 3222)
world = client.get_world()
# world.unload_map_layer(carla.MapLayer.Buildings)
world_map = world.get_map()
blueprint_library = world.get_blueprint_library()
# SHOW_CAM = SHOW_PREVIEW
STEER_AMT = 1.0
im_width = 448
im_height = 448
bird_camera = None
BIRD_CAMERA_POS_Z = 6 
BIRD_CAMERA_POS_X = 0
BIRD_CAMERA_PITCH = -90
a=[]
def process_img(image):
    v = vehicle.get_velocity()
    ms = int(math.sqrt(v.x**2 + v.y**2 + v.z**2))
    if ms >2:
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((im_height, im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        full_camera = i
        a.append(full_camera)
        # print(len(a))
        time.sleep(2/(2*ms+1))

def signed_angle_2d(A, B):
    # Compute the dot product
    dot_product = np.dot(A, B)
    
    # Compute the cross product (in 2D, it's just a scalar)
    cross_product = A[0] * B[1] - A[1] * B[0]
    
    # Compute the signed angle using atan2
    angle_radians = np.arctan2(cross_product, dot_product)
    
    # Convert to degrees (optional)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def vector(v):
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])
sem_cam = blueprint_library.find('sensor.camera.semantic_segmentation')
sem_cam.set_attribute("image_size_x", f"{im_width}")
sem_cam.set_attribute("image_size_y", f"{im_height}")
sem_cam.set_attribute("fov", f"120")
# sem_cam.set_attribute("pitch", f"-15")
vehicle = world.get_actors().filter('*vehicle*')[0]
# raise ValueError(current_waypoint)
waypoint = world_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
current_waypoint = waypoint
route_waypoints = []
route_waypoints.append(current_waypoint)
bin = blueprint_library.find('static.prop.bin')
for x in range(200):
    next_waypoint = random.choice(current_waypoint.next(1.0))
    route_waypoints.append(next_waypoint)
    current_vector = vector(current_waypoint.transform.get_forward_vector())[:2]
    next_vector = vector(next_waypoint.transform.get_forward_vector())[:2]
    turn_angle = signed_angle_2d(current_vector,next_vector)
    print(turn_angle)
    current_waypoint = next_waypoint
    world.try_spawn_actor(bin, carla.Transform(current_waypoint.transform.location))
    time.sleep(1)
raise ValueError("wait")
camera_init_trans = carla.Transform(carla.Location(z=BIRD_CAMERA_POS_Z,x=BIRD_CAMERA_POS_X),
                                rotation=carla.Rotation(pitch=BIRD_CAMERA_PITCH, yaw=0, roll=0))
sensor = world.spawn_actor(sem_cam, camera_init_trans, attach_to=vehicle)

sensor.listen(lambda data: process_img(data))

t1 = time.time()
i = 0
initial_len = len(os.listdir('/media/carla/AVRL/our_ppo/dataset_creation/results'))
while True:
    
    if i<len(a):
        cv2.imwrite(f'/media/carla/AVRL/our_ppo/dataset_creation/results/{i+initial_len}.jpg', a[i])
        print(len(a)-i)
        i+=1
        time.sleep(0.01)
    else:
        time.sleep(0.2)
    
