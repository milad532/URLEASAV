import carla
import random
import time
import os
import numpy as np


def vector(v):
		if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
			return np.array([v.x, v.y, v.z])
		elif isinstance(v, carla.Rotation):
			return np.array([v.pitch, v.yaw, v.roll])

class ScenarioManager:
    """
    Manages the creation of accident traffic scenarios in a CARLA world.
    """

    def __init__(self, world: carla.World):
        """
        Initializes the ScenarioManager with a CARLA world instance.

        Args:
            world (carla.World): The CARLA world to manage.
        """
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []  # Keep track of actors spawned by this manager
        self.actors_location=[]
        self.walker_blueprints = self.blueprint_library.filter("walker.pedestrian.*")
        self.bicycle_blueprints = self.blueprint_library.filter("vehicle.bh.*")
        self.road_block_blueprints = self.blueprint_library.filter("static.prop.streetbarrier")
        self.vehicle_blueprints = self.blueprint_library.filter("vehicle.*")
        
        self.combined_scenario_chance = 1
        
# 
        # self.lane_actor_types = ["walker", "bike", "block"]
        self.lane_actor_types = ["walker", "bike"]
        
    def generate_scenario(self, ego_vehicle: carla.Actor, route_waypoints: list):
        """
        Generates a random traffic scenario based on pre-defined probabilities.
        Args:
            ego_vehicle (carla.Actor): The main vehicle for which the scenario is created.
            route_waypoints (list): A list of carla.Waypoint objects defining the route.
        """
        self.clear_scenario()

        # print(9999999)
        # print(self.actors_name_list)
        if random.random() < self.combined_scenario_chance:
            # print("Spawning a combined scenario 40 meters ahead...")
            self._spawn_combined_scenario(route_waypoints, distance=40)
        return self.actors_name_list,self.actor_list,self.actors_location


    def clear_scenario(self):
        """
        Destroys all actors that were spawned by this manager.
        """
        # print(f"Clearing {len(self.actor_list)} actors...")
        for actor in self.actor_list:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Error destroying actor: {e}")
        self.actor_list = []
        self.actors_name_list = []
        self.actors_location=[]

    def get_waypoint_ahead(self, start_waypoint: carla.Waypoint, distance: float) -> carla.Waypoint:
        """
        Walks along a waypoint route to find a waypoint at a specified distance.
        """
        current_waypoint = start_waypoint
        distance_walked = 0.0
        
        while distance_walked < distance:
            next_waypoints = current_waypoint.next(0.5)
            if not next_waypoints:
                break
            next_waypoint = random.choice(next_waypoints)
            distance_walked += current_waypoint.transform.location.distance(next_waypoint.transform.location)
            current_waypoint = next_waypoint
        
        return current_waypoint
        
    def _spawn_multiple_actors_in_lane(self, start_waypoint: carla.Waypoint, actor_type: str, count: int):
        """
        Spawns a random number of actors of a specific type starting from a given waypoint.
        """
        waypoint = start_waypoint
        for i in range(count):
            spawn_transform = carla.Transform(
                waypoint.transform.location,
                waypoint.transform.rotation
            )
            
            actor = None
            if actor_type == "walker":
                bp = random.choice(self.walker_blueprints)
                spawn_transform.location.z += 1.0
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                actor = self.world.try_spawn_actor(bp, spawn_transform)

            elif actor_type == "bike":
                bp = random.choice(self.bicycle_blueprints)
                # Ensure the bike is not moving by setting its velocity to zero
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                spawn_transform.rotation.yaw += 90
                actor = self.world.try_spawn_actor(bp, spawn_transform)
                if actor:
                    # Apply a control with brake to ensure it stays stationary
                    actor.apply_control(carla.VehicleControl(brake=1.0))
            
            elif actor_type == "block":
                bp = random.choice(self.road_block_blueprints)
                spawn_transform.rotation.yaw += 90
                spawn_transform.location.z += 0.5
                actor = self.world.try_spawn_actor(bp, spawn_transform)
                
            if actor:
                # print(939393939939393)
                self.actors_name_list.append(actor_type)
                self.actor_list.append(actor)
                self.actors_location.append(vector(spawn_transform.location)[:2])
                # print(f"  -> Spawned {actor_type} #{i+1} at {spawn_transform.location}")
            
            # Find a new waypoint slightly ahead to spawn the next actor
            # The spacing is larger for vehicles and blocks to prevent crowding
            # if actor_type in ["vehicle", "block"]:
            #     next_waypoint = waypoint.next(random.uniform(5.0, 10.0))
            # else:
            #     next_waypoint = waypoint.next(random.uniform(2.0, 5.0))

            # if not next_waypoint:
            #     break
            # waypoint = random.choice(next_waypoint)

    def _spawn_combined_scenario(self, route_waypoints: list, distance: float):
        """
        Spawns a combination of actors 40 meters ahead in both lanes.
        """
        start_waypoint = route_waypoints[0]
        
        main_waypoint = self.get_waypoint_ahead(start_waypoint, distance)
        
        if not main_waypoint:
            print(f"Failed to find a waypoint {distance} meters ahead. Skipping scenario generation.")
            return

        opposite_waypoint = main_waypoint.get_left_lane()
        

        main_lane_actor_type = random.choice(self.lane_actor_types)
        num_to_spawn_main = random.randint(1, 3)
        
        # print(f"Spawning {num_to_spawn_main} {main_lane_actor_type}s in main lane at {main_waypoint.transform.location}:")
        self._spawn_multiple_actors_in_lane(main_waypoint, main_lane_actor_type, num_to_spawn_main)
            
        if opposite_waypoint:

            available_types = [t for t in self.lane_actor_types if t != main_lane_actor_type]
            if available_types:
                opposite_lane_actor_type = random.choice(available_types)
                num_to_spawn_opposite = random.randint(1, 3)
                
                # print(f"Spawning {num_to_spawn_opposite} {opposite_lane_actor_type}s in opposite lane at {opposite_waypoint.transform.location}:")
                self._spawn_multiple_actors_in_lane(opposite_waypoint, opposite_lane_actor_type, num_to_spawn_opposite)
            else:
                print("Could not choose an opposite lane actor type different from the main lane.")
        
                