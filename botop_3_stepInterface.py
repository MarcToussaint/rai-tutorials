# %% [markdown]
# # BotOp-3: Step Inferace (env-like) example
# %%
import robotic as ry
import numpy as np
import time

# %% [markdown]
# A minimal IK method (could be replace by just using the Jacobian to translate the delta)
# %%
def mini_IK(C: ry.Config, pos, qHome):
    q = C.getJointState()
    komo = ry.KOMO(C, 1, 1, 0, False) #one phase one time slice problem, with 'delta_t=1', order=0
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q) #cost: close to 'current state'
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome) #cost: close to qHome
    komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [1e2], pos) #constraint: gripper position
    
    ret = ry.NLP_Solver(komo.nlp()) .setOptions(verbose=0, stopEvals=20, stopTolerance=1e-3) .solve()

    if ret.feasible:
        return ret.x - q
    else:
        return np.zeros((q.size))

def mini_JacIK(C: ry.Config, pos, qHome):
    q = C.getJointState()
    y, J = C.eval(ry.FS.position, ['l_gripper'])
    Jinv = J.T @ np.linalg.pinv(J@J.T+1e-3*np.eye(y.size))
    dq = Jinv @ (pos-y) + 0.1*(np.eye(q.size) - Jinv@J) @ (qHome-q)
    return dq

# %% [markdown]
# basic setup
# %%
C = ry.Config()

C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
qHome = C.getJointState()

target = C.addFrame('target', 'table')
target.setShape(ry.ST.marker, [.1])
target.setRelativePosition([0., .3, .3])
pos = target.getPosition()
cen = pos.copy()
C.view()

bot = ry.BotOp(C, useRealRobot=False)

# %% [markdown]
# A basic observe-decide-step loop
# %%
tau_step = .05
lmbda = .2
for t in range(100):
    bot.sync(C, tau_step) #keep the workspace C sync'ed to real/sim, and idle .1 sec
    pos = cen + .98 * (pos-cen) + 0.02 * np.random.randn(3)
    target.setPosition(pos)
    
    # observe
    obs = bot.stepObservation()
    
    # decide
    # dq = mini_IK(C, pos, qHome)
    dq = mini_JacIK(C, pos, qHome)
    
    # step
    # bot.moveTo(ret.x, timeCost=2., overwrite=True)
    bot.stepAction(dq, obs, lmbda, maxAccel=2.)        

bot.home(C)
