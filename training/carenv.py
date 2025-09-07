'''
building new env:

Steering - continous - done
Acceleration - smooth continous
Clean up at completion of episode
Navigation - define a goal to travel a short distance

'''
#
import random
import time
import numpy as np
import math 
import cv2
import gymnasium as gym
from gymnasium import spaces
import carla
import torch
from torchvision import models,transforms
import timm
from torch import nn
import os
from dataset_creation.aae.ae.autoencoder import load_ae
from scenario_manager import ScenarioManager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

num_envs = 1
import logging
import datetime

		






with torch.no_grad():

	auto_encoder = load_ae(str(BASE_DIR / "dataset_creation/aae/logs/ae-32_1733218343_best.pkl")).cuda()

preprocess = transforms.Compose([

	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess2 = transforms.Compose([
	transforms.Resize((448,448)),
])
# encoder = EncodeState(95)
SECONDS_PER_EPISODE = 5.5

N_CHANNELS = 3
HEIGHT = 448
WIDTH = 448
FIXED_DELTA_SECONDS = 0.1
WAYPOINT_SIZE = 0.5

SHOW_PREVIEW = True
def distance_to_line(A, B, p):
		num   = np.linalg.norm(np.cross(B - A, A - p))
		denom = np.linalg.norm(B - A)
		if np.isclose(denom, 0):
			return np.linalg.norm(p - A)
		return num / denom
	
def vector(v):
		if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
			return np.array([v.x, v.y, v.z])
		elif isinstance(v, carla.Rotation):
			return np.array([v.pitch, v.yaw, v.roll])

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

def vector_size(v):
	return np.sqrt(np.sum((v**2)))

def get_drichlet(object_values,actor):
	alphas_table = {'walker':{'walker':4,'bike':1},
					'bike': {'walker':2,'bike':4}
	}
	alpha_dict = alphas_table[actor]
	assert list(alpha_dict.keys())==list(object_values.keys()), "Order of keys should match in object_values and alpha_dict"
	probs = np.random.dirichlet(list(alpha_dict.values()))
	value = np.sum(probs*np.array(list(object_values.values())))
	return value

def obstacle_detector(object_values,vehicle_vector,vehicle_location,walker_locations: list,walker_list: list):
	walker_values_gt = np.array([object_values[actor] for actor in walker_list])
	walker_values_pr = np.array([get_drichlet(object_values,actor) for actor in walker_list])

	vehicle_walker_vectors = np.array([walker_location - vehicle_location for walker_location in walker_locations])
	vehicle_walker_angles = np.array([signed_angle_2d(vehicle_vector,vec) for vec in vehicle_walker_vectors])
	

	vehicle_walker_distance = np.array([vector_size(v) for v in vehicle_walker_vectors])
	close_walkers = vehicle_walker_distance < 8
	front_walkers = np.abs(vehicle_walker_angles)<120
	walkers_mask = front_walkers*close_walkers


	vehicle_walker_angles = vehicle_walker_angles[walkers_mask]*1.5
	vehicle_walker_distance = vehicle_walker_distance[walkers_mask]
	walker_values_gt = walker_values_gt[walkers_mask]
	walker_values_pr = walker_values_pr[walkers_mask]

	obstacle_array_gt = sectors_from_angles(vehicle_walker_angles,walker_values_gt,vehicle_walker_distance,agg="sum")
	obstacle_array_pr = sectors_from_angles(vehicle_walker_angles,walker_values_pr,vehicle_walker_distance,agg="sum")

	return obstacle_array_gt,obstacle_array_pr


import numpy as np

def sectors_from_angles(
    angles_deg,        # shape (N,), angles in [-180, 180]
    values,            # shape (N,), value per detection
    distances,         # shape (N,), distance per detection
    include_mask=None, # shape (N,), True to include detection
    agg="first",       # "first" (closest), "max", "sum"
    fill_value=0
):
	
    angles_deg = np.asarray(angles_deg)
    values = np.asarray(values)
    distances = np.asarray(distances)
    
    if include_mask is not None:
        include_mask = np.asarray(include_mask).astype(bool)
    else:
        include_mask = np.ones_like(angles_deg, dtype=bool)
    

    # Filter
    angles_deg = angles_deg[include_mask]
    # #print(9999999)
    # #print(values , include_mask)
    values = values[include_mask]
    distances = distances[include_mask]
    
    # Handle empty arrays after filtering
    if len(angles_deg) == 0:
        return np.full(12, fill_value, dtype=values.dtype if len(values) > 0 else float)
    
    # Shift [-180,180] → [0,360)
    angles0_360 = (angles_deg + 180) % 360
    
    # Bin indices (12 sectors of 30°)
    # Handle edge case: 360° should map to bin 0, not bin 12
    bins = (angles0_360 // 30).astype(int)
    bins = np.clip(bins, 0, 11)  # Ensure bins are in [0, 11]
    
    if agg == "sum":
        out = np.full(12, fill_value, dtype=values.dtype)
        np.add.at(out, bins, values)
        return out
        
    elif agg == "max":
        out = np.full(12, fill_value, dtype=float)
        # Use a mask to track which bins have been filled
        filled = np.zeros(12, dtype=bool)
        
        for b, v in zip(bins, values):
            if not filled[b] or v > out[b]:
                out[b] = v
                filled[b] = True
        
        return out
        
    elif agg == "first":  # closest object
        out = np.full(12, fill_value, dtype=values.dtype)
        closest_dist = np.full(12, np.inf)
        
        for b, v, d in zip(bins, values, distances):
            if d < closest_dist[b]:
                closest_dist[b] = d
                out[b] = v
                
        return out
        
    else:
        raise ValueError("agg must be 'first', 'max', or 'sum'.")

class World_Manager():
	def __init__(self,town='Town02',port = 4732,rank = 0):
		self.town = town
		self.port = port
		self.rank = rank
		
	
	def start_world(self,env=None):
			found = False
			while found==False:
				try:
					client = carla.Client("localhost", self.port)
					client.set_timeout(5.0)
					self.world = client.load_world(self.town)
					found = True
				except Exception as e:
					print('Why?',e)
			
			settings = self.world.get_settings()
			# self.settings.no_rendering_mode = True
			settings.synchronous_mode = True
			settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
			self.world.apply_settings(settings)
			if env != None:
				env.world = self.world
			return self.world
			




class CarEnv(gym.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	bird_camera = None
	BIRD_CAMERA_POS_Z = 6 
	BIRD_CAMERA_POS_X = 0
	BIRD_CAMERA_PITCH = -90
	OBJECT_VALUES = {"walker":4,
					 "bike":0.01}
	
	# Set up logging (do this once, typically in your __init__ method)
	
	def __init__(self,world_manager,main=True,evaluation=False,imagelog_stride=20):
		super(CarEnv, self).__init__()
		# try:
		#	 os.remove(f"/media/carla/AVRL/our_ppo/mkfifo/tick_{rank}")
		# except:
		#	 pass
		# try:
		#	 os.remove(f"/media/carla/AVRL/our_ppo/mkfifo/clean_{rank}")
		# except:
		#	 pass
		self.evaluation = evaluation
		self.stride = imagelog_stride


		self.walker_locations=[]
		self.actor_list = []
		self.pedestrians = []
		self.id = 0
		self.run_id=0
		self.colision_type = None
		self.object_values=CarEnv.OBJECT_VALUES
		self.done_cooldown = -1
		
		logging.basicConfig(
		filename=f'logs/{"--".join(["_".join([key, str(value)]) for key, value in zip(self.object_values.keys(), self.object_values.values())])}.log',
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
		)
		
		
		self.rank = world_manager.rank
		client = carla.Client('localhost', world_manager.port)
		client.set_timeout(10.0)
		world = client.get_world()
		self.world = world
		# #print(999999939394939493)
		# self.world.wait_for_tick()

		self.scenario_manager = ScenarioManager(self.world)
		self.town = self.world.get_map().name
		self.main = main

		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		
		# self.observation_space = spaces.Box(low=-1.0, high=1.0,
		# 									shape=([1037]), dtype=float)
		# self.observation_space = spaces.Box(low=-1.0, high=1.0,
		# 							shape=([397]), dtype=float)
		self.observation_space = spaces.Box(low=-1.0, high=1.0,
											shape=([48]), dtype=float)

		self.map = self.world.get_map()
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		self.steer_lock_penalty = 0
		self.previous_steer = 0
		self.time_ran = 0
		self.checkpoint_waypoint_index = 0
		self.just_init = True
		self.turn = 0
		self.routes = list()
		self.obssensor_flag_pr=0
		self.obssensor_flag_gt=0
		#camera init
		camera_init_trans = carla.Transform(carla.Location(z=self.BIRD_CAMERA_POS_Z,x=self.BIRD_CAMERA_POS_X),
								rotation=carla.Rotation(pitch=self.BIRD_CAMERA_PITCH, yaw=0, roll=0))
		
		#Track route generation
		for _ in range(100):
			self.route_waypoints = list()
			if self.town == "Town07":
				self.transform = self.map.get_spawn_points()[38] #Town7  is 38 
				self.total_distance = 750
			elif self.town.split('/')[-1] == "Town02":
				spawn_points = list(self.map.get_spawn_points())
				corner_spawn_points = [spawn_points[idx] for idx in [1,2,4,9,16,17,22,28,30,35,37,40,42,43,46,48,51]]
				spawn_points.extend(corner_spawn_points)
				spawn_points.extend(corner_spawn_points)
				random_integer = 2#torch.randint(1, len(corner_spawn_points), (1,)).item()
				self.transform = corner_spawn_points[random_integer] #Town2 is 1
				self.total_distance = int(4000)
			self.vehicle = None
			while self.vehicle is None:
				# #print(2828938234566)
				self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
				#colision sensor
				colsensor = self.blueprint_library.find("sensor.other.collision")
				self.colsensor = self.world.try_spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
				self.actor_list.append(self.colsensor)
				
				# Add proximity sensor (optional - based on your environment setup)
				# self.proximity_sensor = self.blueprint_library.find("sensor.other.obstacle")
				# self.obstacle_sensor = self.world.try_spawn_actor(self.proximity_sensor,camera_init_trans,attach_to=self.vehicle)
				# Obstacle sensor
				obstacle_blueprint = self.blueprint_library.find('sensor.other.obstacle')
				obstacle_blueprint.set_attribute('distance', '20')  # Detection range in meters
				obstacle_blueprint.set_attribute('hit_radius', '0.5')  # Detection radius
				obstacle_blueprint.set_attribute('only_dynamics', 'false')  # Detect static and dynamic objects
				obstacle_transform = carla.Transform(carla.Location(x=1.5, z=1.0))  # Position on the vehicle
				self.obstacle_sensor = self.world.try_spawn_actor(obstacle_blueprint, obstacle_transform, attach_to=self.vehicle)
				self.actor_list.append(self.obstacle_sensor)
				


			self.tick()
			while self.vehicle.get_location().x == 0 and self.vehicle.get_location().y == 0:

				time.sleep(0.01)
				pass
			self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
			current_waypoint = self.waypoint
			self.route_waypoints.append(current_waypoint)
			self.vehicle.destroy()

			for x in range(self.total_distance):
				if self.town == "Town07":
					if x < 650:
						next_waypoint = current_waypoint.next(WAYPOINT_SIZE)[0]
					else:
						next_waypoint = current_waypoint.next(WAYPOINT_SIZE)[-1]
				elif self.town.split('/')[-1] == "Town02":
					next_waypoint = random.choice(current_waypoint.next(WAYPOINT_SIZE))
				else:
					next_waypoint = current_waypoint.next(WAYPOINT_SIZE)[0]
				self.route_waypoints.append(next_waypoint)
				current_waypoint = next_waypoint
			self.routes.append(self.route_waypoints)

	
		# self.fetures=torch.zeros([0,96])
		# client.load_world(self.town.split('/')[-1])
	
	
	def tick(self,i=1):
		for _ in range(i):
			self.world.tick()
	


	def reset(self,seed=None, return_info=False, options=None):
		# print(f'resetting {self.rank}')
		# try:
		#	 os.remove(f"/media/carla/AVRL/our_ppo/mkfifo/done_{self.rank}")
		# except:
		#	 pass
		logging.info(f"Episode Start")
		self.walker_locations=[]
		self.cleanup()
		self.tick()
		
		self.run_id+=1
		# print(1111111111111111)
		self.current_waypoint_index = -1
		self.center_lane_deviation = 0.0
		# Waypoint nearby angle and distance from it
		self.route_waypoints = self.routes[0]#random.choice(self.routes)
		self.steer_lock_penalty = 0
		self.previous_steer = 0
		self.previous_throttle = 0
		self.collision_hist = []
		self.actor_list = []
		self.colision_type = None
		self.done_cooldown = -1
		if self.evaluation and self.run_id%self.stride==0:
			try:
				os.mkdir(f"./image_logs/{len(os.listdir('image_logs'))+1}")
				# os.mkdir(f"/media/carla/AVRL/our_ppo/evaluation_logs/{len(os.listdir('evaluation_logs'))+1}/{self.rank}")
				
			except:
				pass
		else:
			if self.run_id%self.stride==0:
				try:
					os.mkdir(f"./image_logs/{self.run_id}")
					# for rank in range(num_envs):
					#	 os.mkdir(f"/media/carla/AVRL/our_ppo/image_logs/{self.run_id}/{rank}")
				except:
					pass
		
		
		# else:
		# print(2222222222222)
		self.transform = self.route_waypoints[0].transform
		self.transform.location.z += 1
		self.vehicle = None
		self.walker = None
		
		print(f"{len(self.route_waypoints)}")

		while self.vehicle is None:
			# print(2828938234566)
			self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
			# print(f"vehicle spawn point, {self.transform}")
		#Spawn Walker Here
		# for i in range(20,225,50):
		# 	self.spawn_walker_ahead('walker.pedestrian.0001', distance=i)
		self.scenario_actors,_,self.scenario_actors_locations=self.scenario_manager.generate_scenario(self.vehicle, self.route_waypoints)
		# raise ValueError(self.scenario_actors_list)
		self.tick()
		while self.vehicle.get_location().x == 0 and self.vehicle.get_location().y == 0:

			time.sleep(0.01)
			pass
		self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
		# current_waypoint = self.waypoint

		
		# print(33333333333333)
		self.actor_list.append(self.vehicle)
		self.tick(1)
		self.vehicle.add_impulse(self.waypoint.transform.get_forward_vector()*12000)
		self.tick(2)
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.sem_cam.set_attribute("fov", f"120")
		# self.sem_cam.set_attribute("pitch", f"-15")

		# raise ValueError(current_waypoint)
		
		camera_init_trans = carla.Transform(carla.Location(z=self.BIRD_CAMERA_POS_Z,x=self.BIRD_CAMERA_POS_X),
									  rotation=carla.Rotation(pitch=self.BIRD_CAMERA_PITCH, yaw=0, roll=0))
		self.sensor = self.world.try_spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		# print(4444444444444444444)
		

		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.try_spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))


		#Obstacle sensor
		obstacle_blueprint = self.blueprint_library.find('sensor.other.obstacle')
		obstacle_blueprint.set_attribute('distance', '100')  # Detection range in meters
		obstacle_blueprint.set_attribute('hit_radius', '5')  # Detection radius
		obstacle_blueprint.set_attribute('only_dynamics', 'false')  # Detect static and dynamic objects
		obstacle_transform = carla.Transform(carla.Location(x=1.5, z=1.0))  # Position on the vehicle
		self.obstacle_sensor = self.world.try_spawn_actor(obstacle_blueprint, obstacle_transform, attach_to=self.vehicle)
		self.actor_list.append(self.obstacle_sensor)
		# self.obstacle_sensor.listen(lambda event: self.process_obstacle(event))

		self.tick()
		while self.bird_camera is None:
			time.sleep(0.01)
		self.episode_start = time.time()
		self.steering_loop = False
		self.steering_loop_start = None # this is to count time in steering loop and start penalising for long time in steering loop
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		navigation_obs = torch.tensor([kmh/40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		# print(555555555555555555)

		# observation = preprocess(torch.from_numpy(self.bird_camera).permute(2, 0, 1).float())
		
		with torch.no_grad():
			self.features = torch.cat([torch.sigmoid(torch.from_numpy(auto_encoder.encode_from_raw_image(self.bird_camera[:,:,::-1])).flatten().cpu()),navigation_obs])
			# self.features = torch.cat([dino_encoder(observation.unsqueeze(0).cuda()).squeeze(0).cpu(),navigation_obs])
		
		self.initial_location = self.vehicle.get_location()
		self.prev_time1 = 0
		self.time1 = time.time()
		time.sleep(0.15)
		# print(f"features: {self.features}")
		return self.features,{}

	# def second_reset(self,evaluation=False):
		
		
  
	
	def cleanup(self):
	#	 if self.rank == 0:
	#		 search_list = [idx for idx in range(1,num_envs)]
	#		 while search_list!=[]:
	#			 # print(2222222222222)
	#			 # print(search_list)
	#			 for idx in search_list:
	#				 if os.path.exists(f"/media/carla/AVRL/our_ppo/mkfifo/clean_{idx}"):
	#					 search_list.remove(idx)
	#		 print(search_list)
	#			 # time.sleep(0.5)
		for actor in self.actor_list:
			# print("Destroying actor",actor)
			actor.destroy()
		self.actor_list =[]

		# for sensor in self.world.get_actors().filter('*sensor*'):
		# 	sensor.destroy()


		# for actor in self.world.get_actors().filter('*vehicle*'):
		# 	actor.destroy()
	
	def step(self, action):
		self.obssensor_flag_gt,self.obssensor_flag_pr = obstacle_detector(self.object_values,vector(self.vehicle.get_transform().get_forward_vector())[:2],vector(self.vehicle.get_location())[:2],self.scenario_actors_locations,self.scenario_actors)
		self.prev_time1 = self.time1
		self.time1 = time.time()
		# print(f"stepping {self.time1-self.prev_time1}")
		self.id+=1
		# time.sleep(0.1)


		
		self.step_counter +=1
		done = False
		truncated = False
		steer = float(action[0])
		throttle = float(action[1])
		throttle = (throttle+1)/2
		# map steering actions
		self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0 if throttle>0 else 1.0))

		for i in range(1):
			self.tick()
		time2 = time.time()

		# optional - print steer and throttle every 50 steps		
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		angle = 0
		self.location = self.vehicle.get_location()
		# distance_travelled = self.initial_location.distance(self.location)/10

		
		
		waypoint_index = self.current_waypoint_index

		for _ in range(len(self.route_waypoints)):
			# Check if we passed the next waypoint along the route
			next_waypoint_index = waypoint_index + 1
			wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
			dot = np.dot(vector(wp.transform.get_forward_vector())[:2],vector(self.location - wp.transform.location)[:2])
			if dot > 0.0:
				waypoint_index += 1
			else:
				break
		time3 = time.time()
		lenght =0
		moved_forward = False
		if self.current_waypoint_index < waypoint_index:
			moved_forward = True
			lenght = waypoint_index - self.current_waypoint_index
			
		self.current_waypoint_index = waypoint_index
			
		self.current_waypoint = self.route_waypoints[ self.current_waypoint_index	% len(self.route_waypoints)]
		self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
		self.second_next_waypoint = self.route_waypoints[(self.current_waypoint_index+int(2)) % len(self.route_waypoints)]

		  # self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),vector(self.next_waypoint.transform.location),vector(self.location))
		# self.center_lane_deviation += self.distance_from_center
		actor_vector = vector(self.vehicle.get_transform().get_forward_vector())[:2]
		# normalized_actor_vector = actor_vector/(np.linalg.norm(actor_vector))
		next_forward_vector = vector(self.current_waypoint.transform.get_forward_vector())[:2]
		# normalized_forward_vector = forward_vector/np.linalg.norm(forward_vector)
		second_forward_vector = vector(self.second_next_waypoint.transform.get_forward_vector())[:2]
		turn_angle = signed_angle_2d(next_forward_vector,second_forward_vector)
		self.turn = 0 if np.abs(turn_angle) <2 else np.sign(turn_angle)
		angle = np.sqrt(np.abs(1 - np.dot(actor_vector,next_forward_vector)**2))

		# # storing camera to return at the end in case the clean-up function destroys it
		cam = self.bird_camera
		
		# start defining reward from each step
		self.distance_from_center = ((distance_to_line(vector(self.current_waypoint.transform.location),vector(self.next_waypoint.transform.location),vector(self.location))))
		

		reward = 0
		
	
		good_dist = 2.5 if not self.turn else 2.5
		max_dist = 4.0 if not self.turn else 4.0
		if len(self.collision_hist) != 0 or self.distance_from_center > max_dist:
			if self.colision_type in ['walker','bike'] :
				# reward = - self.OBJECT_VALUES[self.colision_type]
				reward = -(self.obssensor_flag_gt[observation_len//2] + self.obssensor_flag_gt[observation_len//2+1])
				self.done_cooldown=0

			else:
				reward = -20
				self.done_cooldown=0
		elif self.distance_from_center > good_dist:
			reward = -0.1
		else:
			reward = ((1-(self.distance_from_center)/good_dist))*(np.min([kmh/30,1])**2)*(1-np.abs(steer)/1.0)#*(0.3-np.min([angle,0.3]))/0.3	


		if self.obssensor_flag_pr.mean()!=0:
			self.obssensor_flag_normalized = (self.obssensor_flag_pr/self.obssensor_flag_pr.mean())
		else:
			self.obssensor_flag_normalized = self.obssensor_flag_pr
		navigation_obs = torch.tensor([kmh/40,throttle,steer,self.turn]+list(self.obssensor_flag_pr))
		# observation = preprocess(torch.from_numpy(cam).permute(2, 0, 1).float())
		with torch.no_grad():
			encoded_image = auto_encoder.encode_from_raw_image(self.bird_camera[:,:,::-1])
			self.features = torch.cat([torch.sigmoid(torch.from_numpy(encoded_image).flatten().cpu()),navigation_obs])
		# 	features = torch.cat([dino_encoder(observation.unsqueeze(0).cuda()).squeeze(0).cpu(),navigation_obs])
		time4 = time.time()


		if (self.run_id%self.stride==0):# and (self.id%3==0):
			parameters = {
				"reward":f"{reward:.4f}",
				"steer":f"{steer:.2f}",
				"Throttle":f"{throttle:.2f}",
				"kmh":f"{kmh:.0f}",
				"Angle": f"{angle:.4f}",
				"distance_from_center":f"{self.distance_from_center:.2f}",
				"Turn":f"{self.turn:.0f}",
				"turn_angle":f"{turn_angle:.2f}",
				"time":f"{time.time()-self.episode_start:.2f}",
				"Waypoint":f"{self.current_waypoint_index}",
				"dist":f"{self.obssensor_flag_pr}"
			}
			font = cv2.FONT_HERSHEY_SIMPLEX
			font_scale = 0.5
			color = (255, 255, 255)  # White text
			thickness = 1
			line_spacing = 20  # Vertical spacing between lines

			# Calculate text positions dynamically
			y_offset = 20  # Initial vertical offset
			img = self.bird_camera.copy()
			img = np.concatenate((img,preprocess2(torch.tensor(auto_encoder.decode(encoded_image)[0]).permute(2,0,1)).permute(1,2,0)),axis=1)
			
			empty_side = np.zeros_like(self.bird_camera)
			img=np.concatenate((img,empty_side),axis=1)

			for i, (title, value) in enumerate(parameters.items()):
				# Define positions for title (left-aligned) and value (right-aligned)
				title_pos = (2*WIDTH+20, y_offset + i * line_spacing)
				value_pos = (3*WIDTH- 20 - cv2.getTextSize(value, font, font_scale, thickness)[0][0], y_offset + i * line_spacing)
				# Write title and value on the image
				cv2.putText(img, title, title_pos, font, font_scale, color, thickness)
				cv2.putText(img, value, value_pos, font, font_scale, color, thickness)
	
			if self.evaluation:
				cv2.imwrite(f'image_logs/{len(os.listdir("image_logs"))}/{self.id}.jpg', img)
			else:
				cv2.imwrite(f'image_logs/{self.run_id}/{self.id}.jpg', img)
		time5 = time.time()
		
		# if len(self.collision_hist) != 0 or self.distance_from_center>max_dist:
		# 	print(self.colision_type,self.collision_hist)
		# 	print(reward)
		# 	self.cleanup()
		# 	logging.info(f"Episode Ended\n")
		# 	done = True
		if self.done_cooldown>0:
			self.done_cooldown = self.done_cooldown - 1
		elif self.done_cooldown==0:
			self.cleanup()
			logging.info(f"Episode Ended\n")
			done = True
		elif self.step_counter*FIXED_DELTA_SECONDS > SECONDS_PER_EPISODE:
			self.cleanup()
			reward = -10
			logging.info(f"Episode Ended\n")
			done = True
			truncated = True
		# elif reward < -1:
		# 	if (self.turn!=0 and reward < -1.5) or self.turn==0:
		# 		self.cleanup()
		# 		logging.info(f"Episode Ended\n")
		# 		done = True


		time6 = time.time()
		# print(f"The times are:\n t1 is {time2-self.time1:.2f}\n t2 is {time3-time2:.2f}\n t3 is {time4-time3:.2f}\n t4 is {time5-time4:.2f}\n t5 is {time6-time5:.2f}")
		# if time6 - time1 > 0.1 and not done and not self.evaluation:
			
		#	 raise ValueError(f"Time taken for each step:\n Time: {time.time()-self.episode_start:.2f} \nt1 {time2-time1:.2f}\n t2 {time3-time2:.2f}\n t3 {time4-time3:.2f}\n t4 {time5-time4:.2f}\n t5 {time6-time5:.2f}")

		self.previous_steer = steer
		self.previous_throttle = throttle
		# print(f"features: {self.features}")
		return self.features, reward/20, done, truncated, {'kmh':kmh}	#curly brackets - empty dictionary required by SB3 format

	def get_waypoint_ahead(self, distance):
		"""
		Walks down self.route_waypoints starting from index 0 and
		returns the first transform whose location is at least
		`distance` meters from the vehicle’s spawn point.
		"""
		return_transform=self.route_waypoints[-1].transform
		
		origin = self.route_waypoints[0].transform.location
		for wp in self.route_waypoints[1:]:
			if wp.transform.location.distance(origin) >= distance:
				return_transform=wp.transform
				return_transform.location.z=return_transform.location.z+1
				return return_transform
		# fallback if route is shorter than `distance`
		return_transform.location.z=return_transform.location.z+1
		return return_transform
	
	def spawn_walker_ahead(self, walker_bp_id, distance=10):
		"""
		Spawns a single walker blueprint `walker_bp_id` at the
		waypoint ~`distance` meters ahead of the vehicle’s spawn.
		"""
		# 1) get the blueprint
		walker_bp = self.blueprint_library.find(walker_bp_id)
		if walker_bp.has_attribute('is_invincible'):
			walker_bp.set_attribute('is_invincible', 'false')
		# 2) compute the transform 10 m ahead
		spawn_transform = self.get_waypoint_ahead(distance)
		# print(f"walker spawn point, {spawn_transform}")
		# 3) spawn
		# raise ValueError(spawn_transform.location - spawn_transform.location)
		
		walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
		
		if walker:
			self.walker_locations.append(vector(spawn_transform.location)[:2])
			self.actor_list.append(walker)
		return walker

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		i = np.array(image.raw_data)
		i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
		self.bird_camera = i
		
		# raise ValueError(self.front_camera)

	def collision_data(self, event):
			"""
			Process collision data from the collision sensor.
			Detects if the vehicle collided with a pedestrian.
			"""
			self.collision_hist.append(event)
			logging.info(f"Collision detected - Event: {event.other_actor.type_id}")
			if event.other_actor and 'walker' in event.other_actor.type_id:
				self.pedestrians.append(event.other_actor)
			if 'walker' in event.other_actor.type_id:
				self.colision_type='walker'
			elif 'bike' in event.other_actor.type_id:
				self.colision_type='bike'
			else:
				self.colision_type = "static"
			print(self.colision_type)
				# Apply collision penalty
				# print(f"Collision detected with {event.other_actor.type_id}!")
	

	# def process_obstacle(self, event):
	# 	"""
	# 	Process obstacle proximity data.
	# 	Adds information about nearby obstacles (like pedestrians).
	# 	"""
	# 	distance = event.distance
	# 	if distance < 5:  # If the obstacle is within 10 meters
	# 		# print(f"Obstacle detected at {distance:.2f} meters.")
	# 		#store the distance or use it in the observation space
	# 		self.pedestrian_distance = distance
	# 		self.obssensor_flag=1
	# 	else:
	# 		self.pedestrian_distance = None
	# 		self.obssensor_flag=0

	# def calculate_reward(self):
	# 	"""
	# 	Calculates reward based on proximity to pedestrians, collisions, and other conditions.
	# 	"""
	# 	reward = 0
		
	# 	# Penalty for collision with pedestrians
	# 	if len(self.pedestrians) > 0:
	# 		reward -= 1  # Large penalty for collision
		
	# 	# Penalty for proximity to pedestrians (e.g., too close)
	# 	if self.pedestrian_distance and self.pedestrian_distance < 3:  # 5 meters is the safe distance threshold
	# 		reward -= 0.5  # Small penacolty for being too close
		
	# 	# Reward for safe driving (not hitting pedestrians)
	# 	if self.pedestrian_distance and self.pedestrian_distance >= 3:
	# 		reward += 1  # Reward for keeping a safe distance
		
		
	# 	return reward
		
	
 
 

