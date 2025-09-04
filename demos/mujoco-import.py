import robotic as ry
import robotic.src.mujoco_io as mj
import sysconfig
from robotic.src.yaml_helper import *

pysite = sysconfig.get_paths()["purelib"]
# file = pysite+"/gymnasium_robotics/envs/assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml"
file = '/home/mtoussai/git/co-Shiping/assets/kitchens/models/MODERN_2_GALLEY.xml'
# file = '/home/mtoussai/git/co-Shiping/assets/kitchens/models/RUSTIC_ONE_WALL_SMALL.xml'

print('=====================', file)
M = mj.MujocoLoader(file, visualsOnly=True)
M.C.view(True)

yaml_write_dict(M.C.asDict(), 'z.yml')
with open('z.g', 'w') as fil:
    fil.write(M.C.write())
