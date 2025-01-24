# This is a minimalistic demo for box grasm
# It ManipulationModelling to compute optimal waypoints and paths (see komo-3-manipulation)
# It uses the PhysX engine behind botop to actually simulate the grasp, using PD gains in the fingers to excert foces
# The focus of this test is how PhysX responds to setting PD and friction parameters of the grasp
# To this end, the lift is pretty fast... we want to potentially force a slip

# [literally translated from c++ test/21-grasp]

import robotic as ry
import numpy as np
import robotic.manipulation as manip
import time

# these are global parameters by which you can influence the friction, grasp force, etc...
ry.params_add({
    'physx/angularDamping': 0.1,
    'physx/defaultFriction': 3.,  #reduce -> slip
    'physx/defaultRestitution': .7, #quit bouncy
    'physx/motorKp': 1000.,
    'physx/motorKd': 100.,
    'botsim/verbose': 0})

C = ry.Config()
C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))

# C.getFrame('l_panda_finger_joint1').setAttribute('motorKp', 100)
# C.getFrame('l_panda_finger_joint1').setAttribute('motorKd', 10)

C.addFrame("obj") \
      .setPosition([-.25,.1,.7]) \
      .setShape(ry.ST.ssBox, [.04,.2,.1,.005]) \
      .setColor([1,.5,0]) \
      .setMass(.1) \
      .setContact(True)

ways = manip.ManipulationModelling()
ways.setup_sequence(C, 2, 1e-2, 1e-1, False, False, False)
ways.grasp_box(1., 'l_gripper', 'obj', 'l_palm', 'x', .02)
ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, scale=[0,0,1e0], target=[0,0,1])
ways.solve(0)
X = ways.komo.getPath()
print(X)
# ways.komo.view(True)

motion1 = ways.sub_motion(0)
motion1.approach([.8,1.], 'l_gripper')
motion1.solve(0)
# motion1.komo.view(True)

motion2 = ways.sub_motion(1)
motion2.solve(0)
# motion2.komo.view(True)

bot = ry.BotOp(C, useRealRobot=False)

# send a spline for execution, and wait til it's done
bot.move(motion1.path, [1.])
bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

# close gripper
bot.gripperMove(ry.ArgWord._left, .0, 1.5)
bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

# send a spline for execution, and wait til it's done
bot.move(motion2.path, [.3]) #fast upward motion
bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

for t in range(10):
    bot.sync(C, .1)

# open gripper
bot.gripperMove(ry.ArgWord._left, +1., .5)
bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

for t in range(10):
    bot.sync(C, .1)
