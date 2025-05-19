#https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
#https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import mujoco.viewer

# import robotic as ry
# C = ry.Config()
# C.addFile(ry.raiPath('../rai-robotModels/panda/panda.g'))
# C.view(True)


static_model = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(static_model)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  while viewer.is_running():
    mujoco.mj_step(m, d)
    viewer.sync()
