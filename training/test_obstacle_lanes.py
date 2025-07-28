import carla
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Global variables for control and obstacle tracking ---
# This will store True if ANY obstacle sensor detects something.
global_obstacle_detected = False
# This dictionary will store a list of actors detected by each sensor
detected_obstacles_by_sensor = {}
# List to keep track of all spawned walkers and their controllers for cleanup
all_actors = []

# --- CARLA Setup ---
client = carla.Client('localhost', 4732)
client.set_timeout(10.0)
town = 'Town02'
world = client.load_world(town)
blueprint_library = world.get_blueprint_library()

# Set synchronous mode for better control over simulation steps
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05 # Fixed time step for simulation
world.apply_settings(settings)

# --- Spawn Vehicle ---
spawn_id=40
vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[spawn_id]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
all_actors.append(vehicle)
print(f"Spawned vehicle at: {spawn_point.location}")

# --- Obstacle Detection Sensors ---
# We'll create multiple obstacle sensors for different "views"
sensor_bps = blueprint_library.find('sensor.other.obstacle')

# Sensor 1: Front-center
sensor_transform_center = carla.Transform(carla.Location(x=2.5, z=1.0))
sensor_center = world.spawn_actor(sensor_bps, sensor_transform_center, attach_to=vehicle)
all_actors.append(sensor_center)
detected_obstacles_by_sensor['center'] = []

# Sensor 2: Front-left
# Adjust Y for left (negative for left in CARLA's default coordinate system relative to vehicle)
sensor_transform_left = carla.Transform(carla.Location(x=2.5, y=-0.5, z=1.0), carla.Rotation(yaw=-10))
sensor_left = world.spawn_actor(sensor_bps, sensor_transform_left, attach_to=vehicle)
all_actors.append(sensor_left)
detected_obstacles_by_sensor['left'] = []

# Sensor 3: Front-right
# Adjust Y for right (positive for right in CARLA's default coordinate system relative to vehicle)
sensor_transform_right = carla.Transform(carla.Location(x=2.5, y=0.5, z=1.0), carla.Rotation(yaw=10))
sensor_right = world.spawn_actor(sensor_bps, sensor_transform_right, attach_to=vehicle)
all_actors.append(sensor_right)
detected_obstacles_by_sensor['right'] = []

# --- Obstacle Callback Function ---
def obstacle_callback_factory(sensor_name):
    def callback(event):
        global global_obstacle_detected
        if event.other_actor is not None and 'walker' in event.other_actor.type_id:
            # Check if this specific obstacle is already recorded by this sensor
            if event.other_actor.id not in [a.id for a in detected_obstacles_by_sensor[sensor_name]]:
                detected_obstacles_by_sensor[sensor_name].append(event.other_actor)
                print(f"Obstacle detected by {sensor_name} sensor: {event.other_actor.type_id} (ID: {event.other_actor.id}), distance: {event.distance:.2f}m")
                global_obstacle_detected = True # Set the global flag if any sensor detects something
        # You might want to clear detected_obstacles_by_sensor[sensor_name] list
        # if you only want to react to new detections, otherwise it will keep accumulating
        # For simplicity, here we just keep accumulating and rely on global_obstacle_detected.
    return callback

sensor_center.listen(obstacle_callback_factory('center'))
sensor_left.listen(obstacle_callback_factory('left'))
sensor_right.listen(obstacle_callback_factory('right'))






# --- Spawning Walkers ---
def spawn_walkers_in_lanes(world, blueprint_library, num_walkers=3):
    walkers_list = []
    walker_controllers_list = []

    # Get a list of all walker blueprints
    walker_bpts = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    # Get the vehicle's current location and its current lane's waypoint
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world.get_map().get_waypoint(vehicle_location)

    # We want to spawn walkers a bit ahead of the car, and in different lanes
    # Let's try to find waypoints for current, left, and right lanes
    potential_waypoints = []

    # Current lane, a bit ahead
    for _ in range(5): # try a few times to get a good waypoint
        next_waypoint = vehicle_waypoint.next(random.uniform(5, 15))[0] # 5-15 meters ahead
        if next_waypoint:
            # Check if it's a driving lane (not sidewalk, parking etc.)
            if next_waypoint.lane_type == carla.LaneType.Driving:
                potential_waypoints.append(next_waypoint)
                break
        else:
            vehicle_waypoint = world.get_map().get_waypoint(vehicle.get_location()) # update waypoint if next fails

    # Left lane
    left_lane_waypoint = vehicle_waypoint.get_left_lane()
    if left_lane_waypoint and left_lane_waypoint.lane_type == carla.LaneType.Driving:
        # Move forward a bit in the left lane
        for _ in range(5):
            next_left_wp = left_lane_waypoint.next(random.uniform(5, 15))[0]
            if next_left_wp:
                if next_left_wp.lane_type == carla.LaneType.Driving:
                    potential_waypoints.append(next_left_wp)
                    break
            else:
                left_lane_waypoint = vehicle_waypoint.get_left_lane() # update if next fails

    # Right lane
    right_lane_waypoint = vehicle_waypoint.get_right_lane()
    if right_lane_waypoint and right_lane_waypoint.lane_type == carla.LaneType.Driving:
        # Move forward a bit in the right lane
        for _ in range(5):
            next_right_wp = right_lane_waypoint.next(random.uniform(5, 15))[0]
            if next_right_wp:
                if next_right_wp.lane_type == carla.LaneType.Driving:
                    potential_waypoints.append(next_right_wp)
                    break
            else:
                right_lane_waypoint = vehicle_waypoint.get_right_lane() # update if next fails

    # Shuffle the potential waypoints to pick from
    random.shuffle(potential_waypoints)

    for i in range(min(num_walkers, len(potential_waypoints))):
        spawn_point_walker = potential_waypoints[i].transform
        spawn_point_walker.location.z += 2.0 # Raise Z to avoid spawning inside the ground

        walker_bp = random.choice(walker_bpts)
        # Set attributes for walker if needed (e.g., is_invincible)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'true')

        walker = world.try_spawn_actor(walker_bp, spawn_point_walker)
        if walker:
            print(f"Spawned walker {i+1} at: {walker.get_location()}")
            walkers_list.append(walker)
            all_actors.append(walker)

            # Spawn and start walker AI controller
            controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
            if controller:
                walker_controllers_list.append(controller)
                all_actors.append(controller)
                controller.start()
                # Make the walker walk randomly (or towards a specific point if desired)
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1 + random.random()) # Random speed between 1 and 2 m/s
        else:
            print(f"Failed to spawn walker {i+1} at {spawn_point_walker.location}")

    return walkers_list, walker_controllers_list

# --- Main Simulation Loop ---
try:
    print("Simulation running... Spawning walkers and testing obstacle detection. Press Ctrl+C to stop.")

    # Spawn walkers after the simulation starts and the world is stable
    # Give some time for the vehicle to move forward before spawning walkers directly in front.
    world.tick() # Advance one tick to apply settings
    time.sleep(2) # Give a moment for the vehicle to initialize
    
    walkers, controllers = spawn_walkers_in_lanes(world, blueprint_library, num_walkers=3)

    simulation_duration = 20 # seconds
    start_time = time.time()

    while time.time() - start_time < simulation_duration:
        world.tick() # Advance the simulation world by one tick

        # Check the global obstacle detected flag
        if not global_obstacle_detected:
            # Set throttle to 1 (full forward)
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        else:
            # Obstacle detected, stop the car
            print("Global obstacle detected! Stopping the car.")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            # You might want to break here or keep the car stopped for a bit
            # For this example, we'll stop and then exit after the loop finishes.
            # If you want it to remain stopped indefinitely once detected, remove the 'break'
            # and modify the loop condition.
            break # Exit the loop after stopping upon first detection

finally:
    print("Cleaning up actors...")
    # Stop all walker controllers
    for controller in controllers:
        if controller.is_alive:
            controller.stop()

    # Stop sensors
    if sensor_center.is_listening:
        sensor_center.stop()
    if sensor_left.is_listening:
        sensor_left.stop()
    if sensor_right.is_listening:
        sensor_right.stop()


    # Destroy all actors
    client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])
    print("Cleaned up actors.")

    # Restore asynchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)