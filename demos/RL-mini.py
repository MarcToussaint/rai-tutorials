#!/usr/bin/env python

import robotic as ry
import gymnasium as gym
import numpy as np
import time
print('ry version:', ry.__version__, ry.compiled())

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
print(torch.__version__)

'''
# this is a title
## a multi title
# 
'''

class RaiGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    tau_env = .05 #a gym environment step corresponds to 0.05 sec
    tau_sim = .01 #the underlying physics simulation is stepped with 0.01 sec (5 sim steps in an env step)
    time = 0.
    steps = 0
    viewSteps = False
    random_reset = False
    action_scale = .05

    def __init__(self, time_limit, withArm, withEEcontrol, render_mode=None, sim_verbose=0):
        self.time_limit = time_limit
        self.render_mode = render_mode
        self.withEEcontrol = withEEcontrol

        self.setup_config(withArm)
        # self.limits = self.C.getJointLimits()
        self.q0 = self.C.getJointState()
        self.X0 = self.C.getFrameState()
        n = self.q0.size

        self.ee_p0 = self.C.getFrame("plate").getPosition()
        self.box = self.C.getFrame('box')

        self.observation_space = gym.spaces.Box(-2., +2., shape=(2*n + 6,), dtype=np.float32)

        if self.withEEcontrol:
            self.action_space = gym.spaces.Box(-1., +1., shape=(3,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(-1., +1., shape=(n,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.sim = ry.Simulation(self.C, ry.SimulationEngine.physx, verbose=sim_verbose)

    def __del__(self):
        del self.sim
        del self.C

    def setup_config(self, withArm):
        self.C = ry.Config()
        if withArm:
            self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
            self.C.setJointState([.007], ['l_panda_finger_joint1']) #only cosmetics
            #the following makes it a LOT simpler
            self.C.setJointState([.0], ['l_panda_joint2'])
            self.C.setJointState([.7], ['l_panda_joint7'])
            gripper = 'l_gripper'
            self.action_space = .2
        else:
            self.C.addFile(ry.raiPath('scenarios/ballFinger.g'))
            self.C.addFrame('table', '', 'shape: ssBox, pose: [0 0. .6], size: [2.1, 2.1, .1, .02], color: [.3, .3, .3], friction: .5')
            gripper = 'finger'
            self.C.getFrame('jointZ').setJoint(ry.JT.hingeZ, limits=[-3,3])
            self.C.getFrame('base').setPosition([.0, .0,1.])
            if self.withEEcontrol:
                self.action_space = .2
            else:
                self.action_space = .05

        box = self.C.addFrame('box') \
            .setShape(ry.ST.ssBox, size=[.1,.1,.1,.005]) .setColor([1,.5,0]) \
            .setPosition([.06,.35,.7]) \
            .setMass(.1) \
            .setAttribute('friction', .5)
        box.setPosition([.5,.1,.7])

        self.C.addFrame('plate', gripper) \
            .setShape(ry.ST.ssBox, size=[.02,.2,.36,.005]) .setColor([.5,1,0]) \
            .setRelativePosition([0,0,-.16]) \
            .setAttribute('friction', .1)

        self.C.addFrame('target') \
            .setShape(ry.ST.marker, size=[.1]) .setColor([0,1,0]) \
            .setPosition([.3,.3,.7])

    def observation_fct(self):
        X, q, Xdot, qdot = self.sim.getState()
        box_pos = X[self.box.ID, :3]
        box_vel = Xdot[self.box.ID, 0]
        observation = np.concatenate((q, qdot, box_pos, box_vel), axis=0)
        return observation

    def reward_fct(self):
        goalDist, _ = self.C.eval(ry.FS.positionDiff, ["box", "target"])
        sigma = .2
        r = 1. - np.tanh(np.linalg.norm(goalDist)/sigma)

        negObjDistance, _ = self.C.eval(ry.FS.negDistance, ["plate", "box"])
        r += negObjDistance[0]
        return r

    def step_velocity(self, velocity):
        steps = int(self.tau_env/self.tau_sim)
        velocity *= np.power(self.steps,.7)
        for s in range(steps):
            self.sim.step(velocity, self.tau_sim, ry.ControlMode.velocity)
            if self.viewSteps:
                self.C.view()
                time.sleep(self.tau_sim)

    def step_qDelta(self, delta):
        sim_steps = int(self.tau_env/self.tau_sim)
        for s in range(sim_steps):
            q = self.sim.get_q()
            self.sim.step(q + self.action_scale*delta, self.tau_sim, ry.ControlMode.position)
            if self.viewSteps:
                self.C.view()
                time.sleep(self.tau_sim)

    def step_eeDelta(self, delta):
        dx, dy, dtheta = self.action_scale * delta
        dtheta *= 2.
        ee = self.C.getFrame("plate")

        komo = ry.KOMO()
        komo.setConfig(self.C, False)
        komo.setTiming(1, 1, 1., 0)

        sim_steps = int(self.tau_env/self.tau_sim)
        for s in range(sim_steps):
            pos = ee.getPosition() + np.array([dx, dy, 0.0])
            pos[2] = self.ee_p0[2]

            quat = ee.getQuaternion()
            w = ry.Quaternion().set(quat).getLog()
            w = [0., 0., w[2]+dtheta]
            quat = ry.Quaternion().setExp(w).asArr()

            komo.clearObjectives()
            komo.addControlObjective([], 0, 1e-1)
            komo.addObjective([], ry.FS.position, ['plate'], ry.OT.sos, [1e2], pos)
            komo.addObjective([], ry.FS.quaternion, ['plate'], ry.OT.sos, [1e2], quat)
            sol = ry.NLP_Solver(komo.nlp())
            sol.setOptions(stopInners=4, damping=1e-4, verbose=0)
            ret = sol.solve()
            # komo.view(True, f'sol{s}')

            self.sim.step(komo.getPath()[0], self.tau_sim, ry.ControlMode.position)
            if self.viewSteps:
                self.C.view()
                time.sleep(self.tau_sim)

    def step(self, action):
        # self.step_velocity(action)
        if self.withEEcontrol:
            self.step_eeDelta(action)
        else:
            self.step_qDelta(action)

        self.steps += 1
        self.time += self.tau_env
        
        observation = self.observation_fct()
        reward = self.reward_fct()
        terminated = False
        truncated = (self.time >= self.time_limit) # terminated and truncated difference is super important
        info = {"no": "additional info"}
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        X = self.X0.copy()
        if self.random_reset:
            X[self.box.ID, :2] += .3 * np.random.randn(2)

        self.time = 0.
        self.sim.resetTime()
        self.sim.setState(X, self.q0)
        self.sim.resetSplineRef()

        observation = self.observation_fct()
        info = {"no": "additional info"}

        if self.render_mode == "human":
            self.C.view(False)

        return observation, info
        
    def rollout(self, pi):
        self.viewSteps=True
        obs, info = self.reset()
        t = 0
        R = 0
        while True:
            action = pi(obs, t)
            obs, reward, terminated, truncated, info = self.step(action)
            R += reward
            t += 1
            # print("reward: ", reward)
            if terminated or truncated:
                break
        print('total return:', R)
        self.viewSteps=False

    def render(self):
        self.C.view(False, f'RaiGym time {self.time} / {self.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()


def main():

    env = RaiGym(time_limit=3., withArm=True, withEEcontrol=True, render_mode='human', sim_verbose=0)

    # env.rollout(lambda obs, t : np.array([1.,-1.,1.]))
    # env.C.view(True)
    # return

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints_endeffectorActions/', name_prefix='rl_model')
    model = SAC("MlpPolicy", env, gamma=0.99, learning_rate=3e-3, verbose=1, tensorboard_log="./tensorboard/")
    # model = PPO("MlpPolicy", env, gamma=0.99, learning_rate=3e-3, n_steps=1024, verbose=1, tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=20_000, callback=checkpoint_callback)

    print('environment steps:', env.steps)
    # model = SAC.load('./checkpoints_endeffectorActions/rl_model_199498_steps.zip', env=env)

    while True:
        env.rollout(lambda obs, t : model.predict(obs, deterministic=False)[0])
        if env.C.view(True) == ord('q'):
            break

if __name__ == '__main__':
    main()
