import robotic as ry
import sysconfig

pysite = sysconfig.get_paths()["purelib"]
file = pysite+"/gymnasium_robotics/envs/assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml"
#file = '/home/mtoussai/git/MuJoCo2Rai/kitchen_dataset/RUSTIC_ONE_WALL_SMALL.xml'

print('=====================', file)
M = ry.MujocoLoader(file, visualsOnly=True)
M.C.view(True)