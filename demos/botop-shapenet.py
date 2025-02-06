import robotic as ry
import numpy as np
import time

def plan(C: ry.Config, usePredefinedGraspFrame=None):
    ways = ry.KOMO_ManipulationHelper()
    ways.setup_sequence(C, 2, 1e-2, 1e-1, False, False, False)
    if usePredefinedGraspFrame is not None:
        ways.komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', usePredefinedGraspFrame], ry.OT.eq, scale=[1e1]) # impose a constraint to reach a predefined grasp
    else:
        ways.grasp_box(1., 'l_gripper', 'obj', 'l_palm', 'x', .02) # otherwise impose more general box grasp constraints
    ways.komo.addObjective([2.], ry.FS.position, ['l_gripper'], ry.OT.eq, scale=[0,0,1e0], target=[0,0,1]) # impose 'some' constaint also on the 2nd frame, here just lift, later, place with other orientation
    ret = ways.solve(1)
    print('grasp costs/feasibilities:', ret) # this provides a metric for how good/feasible the grasp is kinematically; can be use to reject the grasp
    # ways.komo.view(True)
    if not ret.feasible:
        return None, None

    motion1 = ways.sub_motion(0)
    motion1.approach([.8,1.], 'l_gripper') ## this generates the motion to the first waypoint, with the last 20% constrained to be an 'approach' to the grasp
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

    bot.gripperMove(ry.ArgWord._left, .0, 1.5) #fast closing
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    for t in range(5): #wait a bit
        bot.sync(C, .1)

    bot.move(path2, [1.]) #fast upward motion
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

    # setup a scene with robot and shapenet object
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    C.addFrame('grasp') .setShape(ry.ST.marker, [.2])
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKp', 100)
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKd', 10)

    shapenet_file = 'shapenet/models/1061c1b4af7fd99777f4e9e0a7e4c2.shape.h5'
    obj = C.addH5Object('shapenet', shapenet_file, 1)
    obj.setPosition([.2, .3, 1.])
    obj.setQuaternion([1,0,1,0])

    # setup a copy of that scene just for grasp sampling
    CgraspSample = ry.Config()
    CgraspSample.addConfigurationCopy(C)
    CgraspSample.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaFloatingGripper.g"))
    CgraspSample.addFrame('grasp_ref', 'shapenet_pts') .setShape(ry.ST.marker, [.1])
    CgraspSample.selectJointsBySubtree(CgraspSample.getFrame('base'))


    # start bot sim and wait til dropped
    bot = ry.BotOp(C, useRealRobot=False)
    for t in range(20):
        bot.sync(C, .1)


    for k in range(100):
        #copy object pose into other scene and sample grasp
        CgraspSample.getFrame('shapenet').setPose(C.getFrame('shapenet').getPose())
        relGrasp = ry.DataGen.sampleGraspCandidate(CgraspSample, 'shapenet_pts', 'grasp_ref', .2, verbose=0) #change verbosity!
        CgraspSample.view(False, 'sampled grasp')

        #copy the grasp back to scene
        C.getFrame('grasp').setPose(CgraspSample.getFrame('gripper').getPose())
        # C.view(False, 'proposed grasp')

        # plan motion
        path1, path2 = plan(C, usePredefinedGraspFrame='grasp')

        # execute
        if path1 is not None:
            execute(C, bot, path1, path2)
        else:
            print('== fail ==')

    
if __name__ == '__main__':
    main()

