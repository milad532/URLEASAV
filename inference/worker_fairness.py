import carla
import random
import time
from actors import walker_blueprints,bicycle_blueprints,road_block_blueprints
import os
import datetime
import numpy as np
from PIL import Image
import json
import cv2
from collections import defaultdict
import pandas as pd
import socket
import subprocess
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the script with custom settings.")
    # parser.add_argument('--mode', type=str, default="straigt", help='Set the mode (e.g., "straigt" or "change")')
    parser.add_argument('--port', type=int, default=4040, help='Set the port number')
    parser.add_argument('--walker_chance', type=float, default=0.2, help='the cahnce of walker being hit by the car')
    parser.add_argument('--bike_chance', type=float, default=0.5, help='the cahnce of bike being hit by the car')
    parser.add_argument('--block_chance', type=float, default=0.6, help='the cahnce of car crash with block')
    # Parse the arguments
    return parser.parse_args()

def check_port(port=4040):
    """Check if the given port is being used."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('127.0.0.1', port))  # Check the port on localhost
            return True  # The port is already in use
        except socket.error:
            return False  # The port is not in use

def start_carla(port=4040):
    """Start CARLA with the specified command."""
    carla_command = [
                    "/media/external_10T/milad/carla/CARLA_0.9.14/CarlaUE4.sh", 
                    f"-carla-port={port}", 
                    "-prefernvidia", 
                    "-RenderOffScreen", 
                    "-fps=20"
                    ]
    try:
        print("Starting CARLA...")
        subprocess.Popen(carla_command)  # Starts CARLA as a subprocess
        time.sleep(5)
    except Exception as e:
        print(f"Error starting CARLA: {e}")
        sys.exit(1)

class World_Manager():
    def __init__(self,port,walker_chance,bike_chance,block_chance):
        self.port=port
        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town02")
        # settings = self.world.get_settings()
        # settings.synchronous_mode = False
        # settings.fixed_delta_seconds = 0.1
        # self.world.apply_settings(settings)
        self.world.set_weather(carla.WeatherParameters.WetSunset)
        # self.world =self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.actor_list = []
        self.turn=-1
        self.collision_occurred = False
        self.collision_time = None
        self.collision_log_file=''
        self.obstacle_log_file=''
        self.report_time='' 
        self.images_list = []
        self.LaneChange=0
        self.obstacle_data=[]
        self.collision_data=[]
        # Define obstacle types
        self.bike_types = {'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'}
        self.block_types = {'static.prop.trafficwarning'}
        self.walker_pattern = "walker.pedestrian."
        #decisions
        self.block_chance=block_chance
        self.walker_chance=walker_chance
        self.bike_chance=bike_chance
        self.decision_made=False

    def make_decision(self, obstacle_type):
        """
        Make a decision based on the detected obstacle type.
        """
        if obstacle_type in self.block_types:
            # Block detected
            return "go straight" if random.random() < self.block_chance else "change lane"
        
        elif obstacle_type.startswith(self.walker_pattern):
            # Walker detected
            return "go straight" if random.random() < self.walker_chance else "change lane"
        
        elif obstacle_type in self.bike_types:
            # Bike detected
            return "go straight" if random.random() < self.bike_chance else "change lane"
        
        else:
            # Default action if obstacle is unknown
            return "no decision"

    def spawn_actor(self,number,list2spawn,list4points,yaw=0):
        
        point = self.spawn_points[list4points]
        point.rotation.yaw+=yaw
        list_of_points = []
        y_offsets = [(i)/2 for i in range(-number//2,number//2)]
        self.turn = 1
        x_offsets=[(-1)**(i%2) for i in range(-number//2,number//2)]

        for i in range(number):
            list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i],y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))

        for item,point in zip(list2spawn,list_of_points):

            blueprint = self.blueprint_library.find(item)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            self.actor_list.append(self.world.try_spawn_actor(blueprint,point))
    
    def save_image(self,data, folder):
        # Convert CARLA raw image to numpy array
        
        self.images_list.append(data)
        time.sleep(0.01)
        # print(f"Saved: {file_path}")

    def ego_vehicle(self,ego_spawn=40,save_image=True,lane_mode="straight"):
        print(f'spawning ego vehicle')
        vehicle_blueprint = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_spawn_point = self.spawn_points[ego_spawn]

        vehicle = self.world.spawn_actor(vehicle_blueprint, ego_spawn_point)
        print(f"Spawned ego vehicle: {vehicle_blueprint.id} at location {ego_spawn_point.location}")
            # Add RGB camera sensor to the vehicle
        camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute('image_size_x', '800')  # Width
        camera_blueprint.set_attribute('image_size_y', '600')  # Height
        camera_blueprint.set_attribute('fov', '90')  # Field of View
        camera_blueprint.set_attribute('gamma', '2.2')
        camera_blueprint.set_attribute('sensor_tick', '0.05')
        camera_transform = carla.Transform(carla.Location(x=-1, z=2.5))  # Position on the vehicle
        camera = self.world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
        print(f'camera spawned')
        #spawn the collision sensor
        collision_blueprint = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        collision_sensor = self.world.spawn_actor(collision_blueprint, collision_transform, attach_to=vehicle)
        # Add Obstacle Detector
        obstacle_blueprint = self.blueprint_library.find('sensor.other.obstacle')
        obstacle_blueprint.set_attribute('distance', '20')  # Detection range in meters
        obstacle_blueprint.set_attribute('hit_radius', '0.5')  # Detection radius
        obstacle_blueprint.set_attribute('only_dynamics', 'false')  # Detect static and dynamic objects
        obstacle_transform = carla.Transform(carla.Location(x=1.5, z=1.0))  # Position on the vehicle
        obstacle_detector = self.world.spawn_actor(obstacle_blueprint, obstacle_transform, attach_to=vehicle)

        # Set up the folder to save images
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_folder = f"/media/external_10T/milad/carla_project/inference/image_logs/output_images_{timestamp}"
        self.report_time=timestamp
        camera.listen(lambda data: self.save_image(data, save_folder))
        # Log file for collisions /media/external_10T/milad/carla_project/inference/collision_logs
        self.collision_log_file = f"/media/external_10T/milad/carla_project/inference/collision_logs/{timestamp}.txt"
        self.obstacle_log_file = f"/media/external_10T/milad/carla_project/inference/obstacle_logs/{timestamp}.txt"
        # Attach a callback to the collision sensor
        collision_sensor.listen(lambda event: self.collision_callback(event, self.collision_log_file))
        # Attach a callback to the obstacle detector
        obstacle_detector.listen(lambda event: self.obstacle_callback(event, self.obstacle_log_file))

        # # Set constant control values
        control = carla.VehicleControl()
        control.throttle = 0.7  # Maximum throttle
        control.brake = 0.0     # No braking
        vehicle.apply_control(control)
        # self.map = self.world.get_map()
        # waypoint = self.map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        # vehicle.add_impulse(waypoint.transform.get_forward_vector()*100000000)
        while True:
                if lane_mode=="change":
                    if self.LaneChange==1:                        
                        control.steer = -0.2
                        vehicle.apply_control(control)
                        time.sleep(0.5)
                        control.steer = 0.2
                        vehicle.apply_control(control)
                        time.sleep(0.5)
                        control.steer = 0
                        vehicle.apply_control(control)
                        print(f"lane chnaged")
                        self.LaneChange=2
                # velocity = vehicle.get_velocity()
                # speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5 * 3.6  # Convert m/s to km/h
                # print(f"Current speed: {speed:.2f} km/h")
                if self.collision_occurred:
            # Check elapsed time since the collision
                    elapsed_time = time.time() - self.collision_time
                    if elapsed_time >= 0.5:
                        if camera is not None:
                            camera.destroy()  # Stop after 3 seconds
                            collision_sensor.destroy()
                            obstacle_detector.destroy()
                        else:    
                            collision_sensor.destroy()
                            obstacle_detector.destroy()
                        if save_image==True:
                            print(f"Saving images to folder: {save_folder}")
                            for data in self.images_list:
                                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                                array = array.reshape((data.height, data.width, 4))  # BGRA format
                                image = array[:, :, :3]  # Extract RGB channels
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # Create the folder if it doesn't exist
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                            
                            # Generate file name using frame number
                                file_name = f"{data.frame}.png"
                                file_path = os.path.join(save_folder, file_name)

                            # Save image
                                Image.fromarray(image).save(file_path)
                        print("Stopping simulation 1 seconds after collision.")
                        break
        
    
    def collision_callback(self,event, log_file):
        actor = event.other_actor  # The actor the ego vehicle collided with
        collision_type = event.normal_impulse  # Collision impulse (force)

        # Log collision details
        collision_data = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collided_actor_id": actor.id,
            "collided_actor_type": actor.type_id,
            "collision_impulse": {
                "x": collision_type.x,
                "y": collision_type.y,
                "z": collision_type.z,
            },
        }

        # Print to console
        # print(f"Collision detected: {collision_data}")
        if not self.collision_occurred:
            self.collision_time = time.time()
            self.collision_occurred = True
        # Save to log file
        # with open(log_file, "a") as file:
        #     file.write(json.dumps(collision_data) + "\n")
        self.collision_data.append(collision_data)

    def obstacle_callback(self,event, log_file):
        # Extract obstacle details
        obstacle_id = event.other_actor.id if event.other_actor else "None"
        obstacle_type = event.other_actor.type_id if event.other_actor else "None"
        distance = event.distance  # Distance to the detected obstacle

        # Log obstacle details
        obstacle_data = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "obstacle_id": obstacle_id,
            "obstacle_type": obstacle_type,
            "distance": distance,
        }

        # Print to console
        # print(f"Obstacle detected: {obstacle_data}")
        if not self.decision_made:
            decision = self.make_decision(obstacle_type)
            print(f"Decision based on obstacle {obstacle_type}: {decision}")
            if self.LaneChange==0 and decision == "change lane":
                self.LaneChange = 1
            self.decision_made=True
        # Save to log file
        # with open(log_file, "a") as file:
        #     file.write(json.dumps(obstacle_data) + "\n")
        self.obstacle_data.append(obstacle_data)

    def parse_and_generate_report(self,X):
        collision_log_file=self.collision_log_file
        obstacle_log_file=self.obstacle_log_file
        # Define object classifications
        bike_types = {'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'}
        block_types = {'static.prop.trafficwarning'}
        walker_pattern = "walker.pedestrian."
        passenger_number=0
        # Initialize counters and data
        created_objects = {"line 1": defaultdict(int), "line 2": defaultdict(int)}
        detected_objects = set()
        collided_objects = {"walker": 0, "bike": 0, "block": 0, "other": 0}
        collision_data = []
        obstacle_detection_data=[]
        # Parse collision log
        # with open(collision_log_file, "r") as collision_file:
        for line in self.collision_data:
            collided_actor_id = line['collided_actor_id']
            collided_actor_type = line['collided_actor_type']
            if collided_actor_id in [entry['Collided_Actor_ID'] for entry in collision_data]:
                continue
            else:
                collision_data.append({
                "Collided_Actor_ID": collided_actor_id,
                "Collided_Actor_Type": collided_actor_type
                })
                # Skip static objects for collision counting
                if collided_actor_type.startswith("static."):
                    continue
                # Classify the collided object
                if walker_pattern in collided_actor_type:
                    collided_objects["walker"] += 1
                elif collided_actor_type in bike_types:
                    collided_objects["bike"] += 1
                elif collided_actor_type in block_types:
                    collided_objects["block"] += 1
                    passenger_number=1
                else:
                    collided_objects["other"] += 1

       # Parse obstacle detection log
        # with open(obstacle_log_file, "r") as obstacle_file:
        for line in self.obstacle_data:
            obstacle_type = line['obstacle_type']
            obstacle_id=line['obstacle_id']
            if obstacle_id in [entry['obstacle_id'] for entry in obstacle_detection_data]:
                continue
            else:

                obstacle_detection_data.append({
                    "obstacle_id":obstacle_id,
                    "obstacle_type":obstacle_type
                }
                )


        collision_data_df=pd.DataFrame(collision_data)
        obstacle_detection_data_df=pd.DataFrame(obstacle_detection_data)
        detected_count=obstacle_detection_data_df['obstacle_id'].nunique()-1
        unique_collided=collision_data_df['Collided_Actor_ID'].nunique()-1

        # Generate the report
        cost_X1=0
        if(X[0]['item']=='block'):
            cost_X1=(X[0]['number'])*1
        elif(X[0]['item']=='bike'):
            cost_X1=(X[0]['number'])*2
        elif(X[0]['item']=='walker'):
            cost_X1=(X[0]['number'])*3
        
        cost_X2=0
        if(X[1]['item']=='block'):
            cost_X2=(X[1]['number'])*1
        elif(X[1]['item']=='bike'):
            cost_X2=(X[1]['number'])*2
        elif(X[1]['item']=='walker'):
            cost_X2=(X[1]['number'])*3
        
        min_cost = min(cost_X1,cost_X2)
        max_cost = max(cost_X1,cost_X2)
        walker_number=collided_objects['walker']
        bike_number=collided_objects['bike']
        if collided_objects['block']==0 and bike_number ==0 and walker_number==0:
            passenger_number=1
        
        cost=passenger_number*1+collided_objects['bike']*2+collided_objects['walker']*3
        # print(collision_data)
        
        # collision_data_df.to_csv(f"/media/external_10T/milad/carla_project/inference/csv_logs/collision_{self.report_time}.csv", index=False)
        # obstacle_detection_data_df.to_csv(f"/media/external_10T/milad/carla_project/inference/csv_logs/obstacle_{self.report_time}.csv", index=False)
        # print(obstacle_detection_data)
        actors_list=[]
        for item in self.actor_list:
            id=item.id
            type=item.type_id
            position=item.get_location()
            actors_list.append({
                "actor_id":id,
                "actor_type":type,
                "actor_position_x":position.x,
                "actor_position_y":position.y,
                "actor_position_z":position.z
            })
        report = {
            "time":self.report_time,
            "Created X1": X[0]['number'],
            "Created X2": X[1]['number'],
            "Detected": detected_count,
            "Collided": unique_collided,
            "Min Cost": min_cost,
            "Max Cost": max_cost,
            "Cost": cost,
            "walker":walker_number,
            "bike":bike_number,
            "passenger":passenger_number,
            "actors_list":actors_list,
            "collisions":collision_data
        }

        return report




    # def five_spawn(self,list2spawn,list4points):
    #     print(444444)
    #     point = self.spawn_points[list4points]
    #     list_of_points = []
    #     y_offsets=[-1.5,-0.5,0.5,1.5]
    #     x_offsets=[1,-1,1,-1]
    #     for i in range(4):
    #         list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i],y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))
    #     for item,point in zip(list2spawn,list_of_points):
    #         blueprint = self.blueprint_library.find(item)
    #         self.actor_list.append(self.world.spawn_actor(blueprint,point))
            
    # def three_spawn(self,list2spawn,list4points):
    #     print(3333)
    #     point = self.spawn_points[list4points]
    #     list_of_points = []
    #     y_offsets = [-2,0,2]
    #     x_offsets=[1,-1,1]
    #     for i in range(3):
    #         list_of_points.append(carla.Transform(carla.Location(x=point.location.x+x_offsets[i],y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))
    #     for item,point in zip(list2spawn,list_of_points):
    #         blueprint = self.blueprint_library.find(item)
    #         self.actor_list.append(self.world.spawn_actor(blueprint,point))
            
    # def two_spawn(self,list2spawn,list4points):
    #     print(22222)
    #     point = self.spawn_points[list4points]
    #     list_of_points = []
    #     y_offsets = [-1,1]
    #     for i in range(2):
    #         list_of_points.append(carla.Transform(carla.Location(x=point.location.x,y=point.location.y+y_offsets[i],z=point.location.z),point.rotation))
    #     for item,point in zip(list2spawn,list_of_points):
    #         blueprint = self.blueprint_library.find(item)
    #         self.actor_list.append(self.world.spawn_actor(blueprint,point))
    # def one_spawn(self,list2spawn,list4points):
    #     print(1111)
    #     point = self.spawn_points[list4points]
    #     list_of_points = []
    #     for i in range(1):
    #         point.rotation.yaw-=90
    #         list_of_points.append(carla.Transform(carla.Location(x=point.location.x,y=point.location.y,z=point.location.z),point.rotation))
    #     for item,point in zip(list2spawn,list_of_points):
    #         blueprint = self.blueprint_library.find(item)
    #         self.actor_list.append(self.world.spawn_actor(blueprint,point))

plans={
       'plan1':['walker','walker'],
       'plan2':['block','walker'],
       'plan3':['bike','walker'],
       'plan4':['block','bike']
       }

def creaet_scenario(port=4040):
    args = parse_args()
    spawn_point=[34,36]
    ego_spawn=40
    walkers=walker_blueprints
    bikes=bicycle_blueprints
    road_blocks=road_block_blueprints
    walker_chance=args.walker_chance
    bike_chance=args.bike_chance
    block_chance=args.block_chance
    wolrd_class=World_Manager(port=port,
                              walker_chance=walker_chance,
                              bike_chance=bike_chance,
                              block_chance=block_chance)
    
    plan= random.choice(list(plans.values()))
    # raise ValueError(plans.items())
    X=[]
    for j,i in enumerate(plan):
        print(i)
        if(i=='walker'):
            number=random.choice(list(range(1,6)))
            # walker_agents=random.sample(walkers,5)
            wolrd_class.spawn_actor(number,walkers,spawn_point[j])
        elif(i=='bike'):
            number=random.choice(list(range(1,6)))
            # bikes_agents=random.sample(bikes,)
            wolrd_class.spawn_actor(number,bikes,spawn_point[j])
        elif(i=='block'):
            number =1
            block_prop=random.sample(road_blocks,1)
            wolrd_class.spawn_actor(number,block_prop,spawn_point[j],yaw=90)
        print(number)
        X.append({
            "item":i,
            "number":number
            })
        

    #add vehicle
    print(f'scenario added') 
    # List of possible values
    # modes = ["straigt", "change"]
    # mode = random.choice(modes)
    mode="change"
    wolrd_class.ego_vehicle(save_image=False,lane_mode=mode)
    report=wolrd_class.parse_and_generate_report(X)
    # report_df=pd.DataFrame(report,index=[0])
    print(report)
    report_path = "/media/external_10T/milad/carla_project/inference/report/target_exp1_log.txt"
    # existing_report = pd.read_csv(report_path)
    # updated_report = pd.concat([existing_report, report_df], ignore_index=True)
    # updated_report.to_csv(report_path, index=False)
    # report_df.to_csv(report_path, index=False)
    with open(report_path, 'a') as file:  # Open file in append mode
        # Convert the dictionary to a JSON string for better readability in the text file
        file.write(json.dumps(report, indent=4))  # Writes the dictionary as a formatted JSON string
        file.write("\n\n")  # Add a new line to separate entries

    print("report has been updated and saved to the file!")
    

if __name__ == '__main__':
    args = parse_args()
    # Use the values from the arguments
    print(f"args loaded")
    port = args.port
    for i in range(1000):
        while not check_port(port):
            print(f"Port {port} is not in use. Starting CARLA...")
            time.sleep(10)
            start_carla(port=port)
        
        print(f"CARLA is running on port {port}.")
    
        creaet_scenario(port=port)
