import robotic as ry
import numpy as np
import time

def mini_IK(C, pos, qHome):
    q0 = C.getJointState()
    komo = ry.KOMO(C, 1, 1, 0, False) #one phase one time slice problem, with 'delta_t=1', order=0
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome) #cost: close to qHome
    komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [1e2], pos) #constraint: gripper position
    
    ret = ry.NLP_Solver(komo.nlp()) .setOptions(verbose=0, stopEvals=20, stopTolerance=1e-3) .solve()
    
    return ret

def main():
    C = ry.Config()

    C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
    qHome = C.getJointState()

    target = C.addFrame('target', 'table')
    target.setShape(ry.ST.marker, [.1])
    target.setRelativePosition([0., .3, .3])
    pos = target.getPosition()
    cen = pos.copy()
    C.view()

    bot = ry.BotOp(C, useRealRobot=True)

    tau_step = .05
    lmbda = .2
    for t in range(100):
        bot.sync(C, tau_step) #keep the workspace C sync'ed to real/sim, and idle .1 sec
        pos = cen + .98 * (pos-cen) + 0.02 * np.random.randn(3)
        target.setPosition(pos)
        
        ret = mini_IK(C, pos, qHome)
        print(ret)
        if ret.feasible:
            # bot.moveTo(ret.x, timeCost=2., overwrite=True)
           obs = bot.stepObservation()
           bot.stepAction(ret.x - obs.qpos, obs, lmbda, maxAccel=2.)
            

    bot.home(C)

if __name__ == "__main__":
    main()
