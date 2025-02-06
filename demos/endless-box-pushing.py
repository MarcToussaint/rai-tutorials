import robotic as ry
import random

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.addFrame('box', 'table') \
    .setJoint(ry.JT.rigid) \
    .setShape(ry.ST.ssBox, [.15,.06,.06,.005]) \
    .setRelativePosition([-.0,.3-.055,.095]) \
    .setContact(1) \
    .setMass(.1)

C.delFrame('panda_collCameraWrist')

# for convenience, a few definitions:
qHome = C.getJointState()
gripper = 'l_gripper'
palm = 'l_palm'
box = 'box'
table = 'table'
boxSize = C.getFrame(box).getSize()

C.view(True)
C.get_viewer().visualsOnly()

for l in range(20):
    qStart = C.getJointState()

    graspDirection = random.choice(['y', 'z']) #'x' not possible: box too large
    placeDirection = random.choice(['x', 'y', 'z', 'xNeg', 'yNeg', 'zNeg'])
    info = f'placement {l}: grasp {graspDirection} place {placeDirection}'
    print('===', info)

    M = ry.KOMO_ManipulationHelper(info)
    M.setup_pick_and_place_waypoints(C, gripper, box, homing_scale=1e-1)
    M.grasp_box(1., gripper, box, palm, graspDirection)
    M.place_box(2., box, table, palm, placeDirection)
    M.no_collisions([], [palm, table])
    M.target_relative_xy_position(2., box, table, [.2, .3])
    ways = M.solve()

    if not M.feasible:
        continue

    M2 = M.sub_motion(0)
    # M = ry.KOMO_ManipulationHelper(C, info, helpers=[gripper])
    # M.setup_point_to_point_motion(qStart, ways[0])
    M2.no_collisions([.3,.7], [palm, box], margin=.05)
    M2.retract([.0, .2], gripper)
    M2.approach([.8, 1.], gripper)
    M2.solve()
    if not M2.feasible:
        continue

    M3 = M.sub_motion(1)
    #ry.KOMO_ManipulationHelper(C, info)
    # M.setup_point_to_point_motion(ways[0], ways[1])
    M3.no_collisions([], [table, box])
    M3.solve()
    if not M3.ret.feasible:
        continue

    M2.play(C, 1, True)
    C.attach(gripper, box)
    M3.play(C, 1, True)
    C.attach(table, box)

del M