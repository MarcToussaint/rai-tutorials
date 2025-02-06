# This is a minimalistic demo for box grasm
# It ManipulationModelling to compute optimal waypoints and paths (see komo-3-manipulation)
# It uses the PhysX engine behind botop to actually simulate the grasp, using PD gains in the fingers to excert foces
# The focus of this test is how PhysX responds to setting PD and friction parameters of the grasp
# To this end, the lift is pretty fast... we can potentially force a slip when choosing low friction

import robotic as ry
import numpy as np
import time

def plan(C, usePredefinedGrasp):
    ways = ry.KOMO_ManipulationHelper()
    ways.setup_sequence(C, 2, 1e-2, 1e-1, False, False, False)
    if usePredefinedGrasp:
        ways.komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', 'predefined_grasp'], ry.OT.eq, scale=[1e1]) # impose a constraint to reach a predefined grasp
    else:
        ways.grasp_box(1., 'l_gripper', 'obj', 'l_palm', 'x', .02) # otherwise impose more general box grasp constraints
    ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, scale=[0,0,1e0], target=[0,0,1]) # impose 'some' constaint also on the 2nd frame, here just lift, later, place with other orientation
    ret = ways.solve(0)
    print('grasp costs/feasibilities:', ret) # this provides a metric for how good/feasible the grasp is kinematically; can be use to reject the grasp
    # ways.view(True)
    if not ret.feasible:
        return None, None

    motion1 = ways.sub_motion(0)
    motion1.approach([.8,1.], 'l_gripper') ## this generates the motion to the first waypoint, with the last 20% constrained to be an 'approach' to the grasp
    ret = motion1.solve(0)
    # motion1.view(True)
    if not ret.feasible:
        return None, None

    motion2 = ways.sub_motion(1)
    ret = motion2.solve(0)  ## no additional constraints at all on the motion between 1st and 2nd waypoint; later: up and down motion
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

    # this adds a helper frame to specify the predefined grasp, here hardcoded relative to the object; for AnyGrasp perhaps in world (no parent frame) or camera coordinates (camera as parent)
    predefined_grasp = C.addFrame('predefined_grasp', 'obj')
    predefined_grasp.setRelativePosition([.0,.0,.02])  #2cm above center
    predefined_grasp.setShape(ry.ST.marker, [.1])

    for i in range(10):
        # rnd object pose
        obj.setPosition(np.random.uniform([-.5,0.,.7], [.5,.5,.7]))
        obj.setQuaternion(np.random.uniform([-1.,0.,0.,-1.], [1.,0.,0.,1.]))
        C.view()

        # plan
        t = -time.perf_counter()
        path1, path2 = plan(C, usePredefinedGrasp=False)
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

