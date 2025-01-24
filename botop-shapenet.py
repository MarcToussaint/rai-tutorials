# This is a minimalistic demo for box grasm
# It hard-codes the grasp using waypoints (-> should be replaced by model-based grasp planning (see komo-3-manipulation), or pcl-based grasp prediction)
# It uses the PhysX engine behind botop to actually simulate the grasp, using PD gains in the fingers to excert foces
# The focus of this test is how PhysX responds to setting PD and friction parameters of the grasp
# To this end, the lift is pretty fast... we want to potentially force a slip

# [literally translated from c++ test/21-grasp]

import robotic as ry
import numpy as np

# these are global parameters by which you can influence the friction, grasp force, etc...
ry.params_add({
    'physx/angularDamping': 0.1,
    'physx/defaultFriction': 1.,  #reduce -> slip
    'physx/defaultRestitution': .7, #quit bouncy
    'physx/motorKp': 1000.,
    'physx/motorKd': 100.,
    'physx/gripperKp': 1000., #reduce -> slip
    'physx/gripperKd': 100.,
    'botsim/verbose': 0})

C = ry.Config()
C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))

C.addFrame("obj") \
      .setPosition([-.25,.1,.7]) \
      .setShape(ry.ST.ssBox, [.04,.2,.1,.005]) \
      .setColor([1,.5,0]) \
      .setMass(.1) \
      .setContact(True)

way0 = C.addFrame("way0", "obj") .setShape(ry.ST.marker, [.1]) .setRelativePose("t(0 0 .2)")
way1 = C.addFrame("way1", "obj") .setShape(ry.ST.marker, [.1]) .setRelativePose("t(0 .0 .03)")

C.view()

# compute 2 joint space waypoints from these endeff pose waypoints using komo
komo = ry.KOMO()
komo.setConfig(C, True)
komo.setTiming(2., 1, 5., 0)
komo.addControlObjective([], 0, 1e-0)
komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
komo.addObjective([1.], ry.FS.poseDiff, ["l_gripper", "way0"], ry.OT.eq, [1e1])
komo.addObjective([2.], ry.FS.poseDiff, ["l_gripper", "way1"], ry.OT.eq, [1e1])

ret = ry.NLP_Solver() \
    .setProblem(komo.nlp()) \
    .setOptions(stopTolerance=1e-2, verbose=4 ) \
    .solve()
print(ret)
komo.set_viewer(C.get_viewer())
komo.view(True, 'these are the joint space waypoints,\n which are used as control points of the BotOp spline execution')

ways = komo.getPath()

back_ways = np.concatenate([ways[0], C.getJointState()]) .reshape([2, C.getJointDimension()])
print(back_ways)

bot = ry.BotOp(C, useRealRobot=False)
bot.home(C)

# open gripper
bot.gripperMove(ry.ArgWord._left, +1., .5)
bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

# send a spline for execution, and wait til it's done
bot.move(ways, [2., 3.])
bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

# close gripper
bot.gripperMove(ry.ArgWord._left, .015, .5)
bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

# send a spline for execution, and wait til it's done
bot.move(back_ways, [.1, .5]) #very fast upward motion!
bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

# open gripper
bot.gripperMove(ry.ArgWord._left, +1., .5)
bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

# wait for keypress
bot.wait(C, forKeyPressed=True, forTimeToEnd=False)

# print all params used, esp PhysX related:
ry.params_print()
