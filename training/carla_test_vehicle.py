import carla
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from actors import walker_blueprints,bicycle_blueprints,road_block_blueprints

# Connect to CARLA
client = carla.Client('localhost', 4732)
client.set_timeout(10.0)
town = 'Town02'
world = client.load_world(town)
blueprint_library = world.get_blueprint_library()
#world_variables
actor_list=[]
walkers=walker_blueprints
walker_spawn_point=[34,36]
# Spawn a vehicle
vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]  # Choose your vehicle
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
ego_spawn=40
spawn_point = spawn_points[ego_spawn]

vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Attach an obstacle detection sensor 
sensor_bp = blueprint_library.find('sensor.other.obstacle')
sensor_bp.set_attribute('distance', '10')     # max detection distance in meters
sensor_bp.set_attribute('hit_radius', '0.25')     # sensitivity
sensor_bp.set_attribute('only_dynamics', 'false')  # detect static objects too
sensor_bp.set_attribute('debug_linetrace', 'true')  # enables the debug trace
sensor_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
sensor = world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
obstacle_detected = False
# Attach an obstacle detection sensor left 
left_sensor_bp = blueprint_library.find('sensor.other.obstacle')
left_sensor_bp.set_attribute('distance', '10')     # max detection distance in meters
left_sensor_bp.set_attribute('hit_radius', '0.25')     # sensitivity
left_sensor_bp.set_attribute('only_dynamics', 'true')  # detect static objects too
left_sensor_bp.set_attribute('debug_linetrace', 'true')  # enables the debug trace
left_sensor_transform = carla.Transform(carla.Location(x=2.5, y=-1.0 ,z=1.0))
left_sensor = world.spawn_actor(left_sensor_bp, left_sensor_transform, attach_to=vehicle)

# Attach a LiDAR sensor
# lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
# lidar_bp.set_attribute('range', '50')             # Max detection range in meters
# lidar_bp.set_attribute('rotation_frequency', '10') # Rotations per second
# lidar_bp.set_attribute('channels', '32')          # Number of vertical channels
# lidar_bp.set_attribute('points_per_second', '56000') # Number of points generated per second
# lidar_bp.set_attribute('upper_fov', '10')         # Upper field of view in degrees
# lidar_bp.set_attribute('lower_fov', '-30') 

# lidar_transform = carla.Transform(carla.Location(x=0, z=2.5)) # Position of the LiDAR on the vehicle
# lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# Define a callback for obstacle detection
def obstacle_callback(event):
    
    if event.other_actor is not None:
        if event.distance > 1:
            print(f"Sensor:{event.actor}, Event time:{event.timestamp}, Obstacle detected: {event.other_actor.type_id}, distance: {event.distance:.2f}m")
        # raise ValueError(event.distance)
        # obstacle_detected = True
        # vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        # print(vehicle.controtl)

def lidar_callback(data):
    x=1
    # points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    # x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # sensor_location=lidar_transform.location
    # for px, py, pz in zip(x[::500], y[::500], z[::500]):  
    #         end = carla.Location(x=px + sensor_location.x, y=py + sensor_location.y, z=pz + sensor_location.z)
    #         # Use world.debug.draw_line to draw a temporary line
    #         world.debug.draw_line(lidar_transform, lidar_transform, thickness=0.02, color=carla.Color(255, 0, 0), life_time=0.1)
# lidar_sensor.listen(lidar_callback)

sensor.listen(obstacle_callback)
left_sensor.listen(obstacle_callback)

def spawn_actor(number,list2spawn,list4points,yaw=0):
    
    point = spawn_points[list4points]
    point.rotation.yaw+=yaw
    list_of_points = []
    y_offsets = [(i)/2 for i in range(-number//2,number//2)]
    # self.turn = 1
    x_offsets=[(-1)**(i%2) for i in range(-number//2,number//2)]

    for i in range(number):
        list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i],y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))

    for item,point in zip(list2spawn,list_of_points):

        blueprint = blueprint_library.find(item)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        actor_list.append(world.try_spawn_actor(blueprint,point))


def spawn_actor_sidewalk(number,list2spawn,list4points,leftright,yaw=0):
    
    point = spawn_points[list4points]
    point.rotation.yaw+=yaw
    point.location.y+=(leftright)*3
    list_of_points = []
    y_offsets = [(i)/2 for i in range(-number//2,number//2)]
    # self.turn = 1
    x_offsets=[(-1)**(i%2) for i in range(-number//2,number//2)]

    for i in range(number):
        list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i],y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))

    for item,point in zip(list2spawn,list_of_points):

        blueprint = blueprint_library.find(item)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        actor_list.append(world.try_spawn_actor(blueprint,point))


# Spawn walkers
spawn_actor(4,walkers,walker_spawn_point[0])
spawn_actor(5,walkers,walker_spawn_point[1])
spawn_actor_sidewalk(4,walkers,walker_spawn_point[0],-1)
spawn_actor_sidewalk(5,walkers,walker_spawn_point[1],1)

# Attach a LiDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('points_per_second', '56000')
lidar_bp.set_attribute('upper_fov', '10')
lidar_bp.set_attribute('lower_fov', '-30')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# LiDAR visualization and save
frame_num = 0
os.makedirs('lidar_images', exist_ok=True)

def save_lidar_image(lidar_data, frame=0):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Save 2D top-down image
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=z, s=0.1, cmap='viridis')
    plt.axis('off')
    plt.title(f'LiDAR Frame {frame}')
    plt.savefig(f'lidar_images/lidar_frame_{frame:04d}.png', bbox_inches='tight')
    plt.close()

    # Optional: draw rays as debug lines in world
    sensor_location = lidar_sensor.get_transform().location
    for px, py, pz in zip(x[::200], y[::200], z[::200]):  # sparse points to reduce clutter
        end = carla.Location(x=px + sensor_location.x, y=py + sensor_location.y, z=pz + sensor_location.z)
        world.debug.draw_line(sensor_location, end, thickness=0.02, color=carla.Color(255, 0, 0), life_time=0.1)

def lidar_callback(data):
    global frame_num
    # save_lidar_image(data, frame=frame_num)
    frame_num += 1

lidar_sensor.listen(lidar_callback)

# Traffic manager setup (not enabling autopilot here)
tm = client.get_trafficmanager(8000)

# Run simulation for a fixed time
try:
    print("Simulation running... Press Ctrl+C to stop.")
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))
    # while True:
    #     if not obstacle_detected:
    #         # Set throttle to 1 (full forward)
    #         vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))
    #     else:
    #         # Obstacle detected, stop the car
    #         print("Obstacle detected! Stopping the car.")
    #         vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    #         break # Exit the loop after stopping
        # world.wait_for_tick() # Advance the simulation world by one tick
    time.sleep(600)

    


finally:
    sensor.stop()
    lidar_sensor.stop()
    sensor.destroy()
    lidar_sensor.destroy()
    vehicle.destroy()
    for walker in actor_list:
        walker.destroy()
    
    print("Cleaned up actors.")