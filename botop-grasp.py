# This is a minimalistic demo for box grasm
# It ManipulationModelling to compute optimal waypoints and paths (see komo-3-manipulation)
# It uses the PhysX engine behind botop to actually simulate the grasp, using PD gains in the fingers to excert foces
# The focus of this test is how PhysX responds to setting PD and friction parameters of the grasp
# To this end, the lift is pretty fast... we can potentially force a slip when choosing low friction

import robotic as ry
import numpy as np
from manipulation import KOMO_ManipulationHelper
import time

def plan(C):
    ways = KOMO_ManipulationHelper()
    ways.setup_sequence(C, 2, 1e-2, 1e-1, False, False, False)
    ways.grasp_box(1., 'l_gripper', 'obj', 'l_palm', 'x', .02)
    ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, scale=[0,0,1e0], target=[0,0,1])
    ret = ways.solve(0)
    # ways.view(True)
    if not ret.feasible:
        return None, None

    motion1 = ways.sub_motion(0)
    motion1.approach([.8,1.], 'l_gripper')
    ret = motion1.solve(0)
    # motion1.view(True)
    if not ret.feasible:
        return None, None

    motion2 = ways.sub_motion(1)
    ret = motion2.solve(0)
    # motion2.view(True)
    if not ret.feasible:
        return None, None

    return motion1.path, motion2.path

def execute(C, bot, path1, path2):
    bot.move(path1, [1.])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.gripperMove(ry.ArgWord._left, .0, 1.5) #fast closing
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.move(path2, [.3]) #fast upward motion
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # for t in range(5): #wait a sec
        # bot.sync(C, .1)

    bot.gripperMove(ry.ArgWord._left, +1., .5) #normal opening
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    # for t in range(5):
        # bot.sync(C, .1)


def main():
    # these are global parameters that influence friction, etc...
    ry.params_add({
        'physx/angularDamping': 0.1,
        'physx/defaultFriction': 3.,  #reduce -> slip
        'physx/defaultRestitution': .7, #quit bouncy
        'physx/motorKp': 1000.,
        'physx/motorKd': 100.,
        'botsim/hyperSpeed': 1.,
        'botsim/verbose': 0}) #1 to see simulation display (not just sync'ed config); 4 to see physx internal model

    # setup a configuration:
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKp', 100)
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKd', 10)
    obj = C.addFrame('obj')
    obj.setPosition([-.25,.1,.7]) \
        .setShape(ry.ST.ssBox, [.04,.2,.1,.005]) \
        .setColor([1,.5,0]) \
        .setMass(.1) \
        .setContact(True)
    
    for i in range(10):
        # rnd object pose
        obj.setPosition(np.random.uniform([-.5,0.,.7], [.5,.5,.7]))
        obj.setQuaternion(np.random.uniform([-1.,0.,0.,-1.], [1.,0.,0.,1.]))
        C.view()

        # plan
        t = -time.perf_counter()
        path1, path2 = plan(C)
        t += time.perf_counter()
        print('planning time: ', t)

        # execute
        if path1 is not None:
            #start a new robot sim each time (could do that in real)
            bot = ry.BotOp(C, useRealRobot=False)
            execute(C, bot, path1, path2)
        else:
            print('== fail ==')

    
if __name__ == '__main__':
    main()

