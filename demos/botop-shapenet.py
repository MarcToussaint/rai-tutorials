# you need to download the shapenet models and grasps:
# see README.md at https://github.com/MarcToussaint/rai-robotModels/tree/master/shapenet

import robotic as ry
import numpy as np
import time
import h5py

def plan(C: ry.Config, usePredefinedGraspFrame=None):
    ways = ry.KOMO_ManipulationHelper()
    ways.setup_sequence(C, 2, 1e-2, 1e-1, True, False, False)
    if usePredefinedGraspFrame is not None:
        ways.komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', usePredefinedGraspFrame], ry.OT.eq, scale=[1e1]) # impose a constraint to reach a predefined grasp
    else:
        ways.grasp_box(1., 'l_gripper', 'obj', 'l_palm', 'x', .02) # otherwise impose more general box grasp constraints
    ways.no_collisions([], ['table', 'l_palm', 'table', 'l_panda_coll7', 'table', 'l_panda_coll6', 'table', 'l_panda_coll5'])
    # ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, scale=[0,0,1e0], target=[0,0,1.1]) # impose 'some' constaint also on the 2nd frame, here just lift, later, place with other orientation
    ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, [], target=[0,.4,1.]) # impose 'some' constaint also on the 2nd frame, here just lift, later, place with other orientation
    ret = ways.solve(1)
    print('grasp costs/feasibilities:', ret) # this provides a metric for how good/feasible the grasp is kinematically; can be use to reject the grasp
    # ways.komo.view(True)
    if not ret.feasible:
        return None, None

    motion1 = ways.sub_motion(0)
    motion1.approach([.8,1.], 'l_gripper') ## this generates the motion to the first waypoint, with the last 20% constrained to be an 'approach' to the grasp
    motion1.no_collisions([], ['l_gripper', 'table'])
    ret = motion1.solve(1)
    # motion1.komo.view(True)
    if not ret.feasible:
        return None, None

    motion2 = ways.sub_motion(1)
    ret = motion2.solve(1)  ## no additional constraints at all on the motion between 1st and 2nd waypoint; later: up and down motion
    # motion2.komo.view(True)
    if not ret.feasible:
        return None, None

    return motion1.path, motion2.path

def execute(C, bot: ry.BotOp, path1, path2):
    bot.move(path1, [1.])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # for t in range(5): #wait a bit
        # bot.sync(C, .1)

    bot.gripperMove(ry.ArgWord._left, .0, 1.5) #fast closing
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    for t in range(5): #wait a bit
        bot.sync(C, .1)

    bot.move(path2, [1.]) #fast upward motion
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    for t in range(5):
        bot.sync(C, .1)

    bot.gripperMove(ry.ArgWord._left, +1., .5) #normal opening
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    for t in range(10):
        bot.sync(C, .1)


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

    # setup a scene with robot
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    C.addFrame('wall1', 'table') .setShape(ry.ST.ssBox, [1.,.1,.1,.01]) .setRelativePosition([0.,.0,.1])
    C.addFrame('wall2', 'table') .setShape(ry.ST.ssBox, [1.,.1,.1,.01]) .setRelativePosition([0.,.7,.1])
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKp', 100)
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKd', 10)

    # add the shapenet obj
    id = '1061c1b4af7fd99777f4e9e0a7e4c2'
    obj = C.addH5Object('shapenet', f'shapenet/models/{id}.shape.h5', 1)
    obj.setPosition([.2, .3, 1.])
    obj.setQuaternion([1,0,1,0])
    obj.setMass(1.) #rescales also inertia matrix

    # add a grasp reference relative to object
    C.addFrame('grasp', 'shapenet_pts') .setShape(ry.ST.marker, [.2])

    # load grasps
    with h5py.File(f'shapenet/grasps/{id}.grasps.h5', 'r') as fil:
        grasps = fil['grasps/success'][()]
    print('loaded grasps:', grasps.shape)
    C.view(True)

    # setup a copy of that scene just for grasp sampling
    CgraspSample = ry.Config()
    CgraspSample.addConfigurationCopy(C)
    CgraspSample.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaFloatingGripper.g"))
    CgraspSample.addFrame('grasp_ref', 'shapenet_pts') .setShape(ry.ST.marker, [.3])
    CgraspSample.selectJointsBySubtree(CgraspSample.getFrame('base'))

    # start bot sim and wait til dropped
    bot = ry.BotOp(C, useRealRobot=False)
    for t in range(20):
        bot.sync(C, .1)

    for i in range(grasps.shape[0]):
        # set grasp reference to grasp
        relpose = grasps[i, :]
        C.getFrame('grasp').setRelativePose(relpose)
        C.view(False)

        # plan motion
        path1, path2 = plan(C, usePredefinedGraspFrame='grasp')

        # execute
        if path1 is not None:
            execute(C, bot, path1, path2)
        else:
            print('== fail ==')

    
if __name__ == '__main__':
    main()

