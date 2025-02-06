import robotic as ry
import numpy as np

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

C.getFrame('l_panda_finger_joint1').setJointState(np.array([.01]))

obj = box
C.getFrame(obj).setRelativePosition([-.0,.3-.055,.095])
C.getFrame(obj).setRelativeQuaternion([1.,0,0,0])

for i in range(20):
    qStart = C.getJointState()

    info = f'push {i}'
    print('===', info)

    M = ry.KOMO_ManipulationHelper(info)
    M.setup_sequence(C, 2, 1e-1, accumulated_collisions=False)
    M.komo.addFrameDof('obj_trans', table, ry.JT.transXY, False, obj) #a permanent moving(!) transXY joint table->trans, and a snap trans->obj
    M.komo.addRigidSwitch(1., ['obj_trans', obj])
    pushStart = M.straight_push([1.,2.], obj, gripper, table)
    #random target position
    M.komo.addObjective([2.], ry.FS.position, [obj], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), .4*np.random.rand(3) - .2+np.array([.0,.3,.0]))
    M.solve()
    if not M.ret.feasible:
        continue

    M1 = M.sub_motion(0, accumulated_collisions=False)
    M1.retractPush([.0, .15], gripper, .03)
    M1.approachPush([.85, 1.], gripper, .03)
    M1.no_collisions([.15,.85], [obj, 'l_palm'], .02)
    M1.no_collisions([.15,.85], [obj, 'l_finger1'], .02)
    M1.no_collisions([.15,.85], [obj, 'l_finger2'], .02)
    M1.no_collisions([], [table, 'l_palm'], .0)
    M1.no_collisions([], [table, 'l_finger1'], .0)
    M1.no_collisions([], [table, 'l_finger2'], .0)
    M1.solve()
    if not M1.ret.feasible:
        continue

    M2 = M.sub_motion(1, accumulated_collisions=False)
    #M2.komo.addObjective([], ry.FS.positionRel, [gripper, pushStart], ry.OT.eq, 1e1*np.array([[1,0,0],[0,0,1]]))
    #move1->komo->addObjective({}, FS_poseRel, {gripper, obj}, OT_eq, {1e1}, {}, 1); //constant relative pose! (redundant for first switch option)

    M2.solve()
    if not M2.ret.feasible:
        continue

    M1.play(C, 1., True)
    C.attach(gripper, obj)
    M2.play(C, 1., True)
    C.attach(table, obj)

del M