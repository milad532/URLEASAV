# import carla
# import random
# import time
# from utils import server_utils
# carla_sh_path='/media/carla/AVRL/carla/CarlaUE4.sh'
# server_manager = server_utils.CarlaServerManager(carla_sh_path, port=2000)
# server_manager.start()

# def main():
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)
#     # version = carla.__version__
#     # print("Carla API version:", version)
#     # raise ValueError(client)
#     world = client.get_world()
#     settings = world.get_settings()
#     settings.synchronous_mode = True
#     world.apply_settings(settings)

#     blueprint_library = world.get_blueprint_library()
    
#     vehicle_bp = blueprint_library.filter('vehicle.*')[0]
#     spawn_point = random.choice(world.get_map().get_spawn_points())
#     vehicle = world.spawn_actor(vehicle_bp, spawn_point)

#     camera_bp = blueprint_library.find('sensor.camera.rgb')
#     camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
#     camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

#     def save_image(image):
#         image.save_to_disk('_out/%06d.png' % image.frame)

#     camera.listen(lambda image: save_image(image))

#     try:
#         for _ in range(100):
#             world.tick()
#             time.sleep(0.05)
#     finally:
#         camera.stop()
#         camera.destroy()
#         vehicle.destroy()

# if __name__ == '__main__':
#     main()


import carla
import random
import time
from utils import server_utils

# carla_sh_path = '/media/carla/AVRL/CARLA_0.9.13_safebench/CarlaUE4.sh'
# server_manager = server_utils.CarlaServerManager(carla_sh_path, port=4000)
# server_manager.start()

def main():
    client = carla.Client('localhost', 2000)
    print(client)
    # client.set_timeout(10.0)
    
    # Check and print the server version
    try:
        server_version = client.get_server_version()
        print("CARLA Server version:", server_version)
    except RuntimeError as e:
        print("Runtime error when getting server version:", e)
        return
    
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def save_image(image):
        image.save_to_disk('_out/%06d.png' % image.frame)

    camera.listen(lambda image: save_image(image))

    try:
        for _ in range(100):
            world.tick()
            time.sleep(0.05)
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
