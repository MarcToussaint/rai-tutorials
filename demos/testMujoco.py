# https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import pprint
import numpy as np
import time

import robotic as ry

C = ry.Config()
# C.addFile(ry.raiPath('../rai-robotModels/panda/panda.g'))
C.addFile("/home/mtoussai/git/playground/22-random/twoFingers.yml")
# dict = C.asDict(False)
# pprint.pp(dict)


class MujocoWriter:
    joint_map = {
        "hingeX": ("hinge", "1 0 0"),
        "hingeY": ("hinge", "0 1 0"),
        "hingeZ": ("hinge", "0 0 1"),
        "transX": ("slide", "1 0 0"),
        "transY": ("slide", "0 1 0"),
        "transZ": ("slide", "0 0 1"),
        "quatBall": ("ball", None),
        "free": ("free", None),
    }
    shape_map = {
        "ssBox": ("box"),
        "capsule": ("capsule"),
        "sphere": ("sphere"),
    }

    def __init__(self, C: ry.Config):
        self.root = ET.Element("mujoco", {"model": "ry_convert"})

        self.default = ET.SubElement(self.root, "default")
        a = ET.SubElement(self.default, "default", {"class": "ryjoint"})
        b = ET.SubElement(a, "position", {"forcerange": "-150 150", "kp": "1000", "kv": "10", "ctrlrange": "-10 10"})

        self.asset = ET.SubElement(self.root, "asset")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.worldbody = ET.SubElement(self.root, "worldbody")

        q0 = C.getJointState()
        # C.setJointState(np.zeros(len(q0)))
        # add all frames without parent, or with a free joint:
        for f in C.getFrames():
            spec = f.asDict()
            isFree = "joint" in spec and spec["joint"] == "free"
            if f.getParent() == None or isFree:
                self.addFrame(f, self.worldbody)
        C.setJointState(q0)

    def as_str(self, input_floats):
        return " ".join([str(f) for f in input_floats])

    def file_as_str(self, filename):
        return filename.replace("<", "").replace(">", "")

    def addFrame(self, f: ry.Frame, parent: ET.Element):
        spec = f.asDict()
        print(f.name, spec)

        d = {"name": f.name}
        if "pose" in spec and "joint" not in spec:
            pose = spec["pose"]
            if len(pose) == 7:
                d["pos"] = self.as_str(pose[:3])
                d["quat"] = self.as_str(pose[3:])
            elif len(pose) == 3:
                d["pos"] = self.as_str(pose)
            elif len(pose) == 4:
                d["quat"] = self.as_str(pose)
            else:
                raise Exception("mal-formed pose")
        a = ET.SubElement(parent, "body", d)

        # is free (in physx convention)
        if "mass" in spec and parent is self.worldbody and "joint" not in spec:
            j = ET.SubElement(a, "freejoint", {})

        # has a joint
        if "joint" in spec:
            if spec["joint"] == "free":
                j = ET.SubElement(a, "freejoint", {})
            else:
                type = self.joint_map[spec["joint"]]

                # create a joint
                mj_args = {"name": f.name, "type": type[0]}
                if type[1] is not None:
                    mj_args["axis"] = type[1]
                for k, v in spec.items():
                    if "mj_joint_" in k:
                        mj_args[k.replace("mj_joint_", "")] = v
                j = ET.SubElement(a, "joint", mj_args)

                # create a motor
                mj_args = {"name": f.name, "joint": f.name, "class": "ryjoint"}
                for k, v in spec.items():
                    if "mj_actuator_" in k:
                        mj_args[k.replace("mj_actuator_", "")] = v
                m = ET.SubElement(self.actuator, "position", mj_args)

        # has a geometry
        geom = None
        if "mesh" in spec:
            name = f"{f.name}_mesh"
            filename = self.file_as_str(spec["mesh"])
            if filename[-2:] == "h5":
                filename = filename[:-2] + "stl"
            m = ET.SubElement(self.asset, "mesh", {"name": name, "file": filename})
            geom = ET.SubElement(a, "geom", {"type": "mesh", "mesh": name})
        elif "shape" in spec:
            # pass
            col = spec["color"]
            if not ((len(col) == 2 or len(col) == 4) and col[-1] < 1):
                type = spec["shape"]
                size = spec["size"]
                if type == "ssBox":
                    geom = ET.SubElement(a, "geom", {"type": "box", "size": self.as_str([0.5 * x for x in size[:3]])})
                elif type == "capsule":
                    geom = ET.SubElement(a, "geom", {"type": "capsule", "size": self.as_str([size[1], 0.5 * size[0]])})
                elif type == "sphere":
                    geom = ET.SubElement(a, "geom", {"type": "sphere", "size": self.as_str([size[0]])})
                else:
                    raise Exception(f"can't convert object of type {type}")

        # has color
        if "color" in spec and geom is not None:
            if len(col) == 4:
                geom.set("rgba", self.as_str(spec["color"]))
            elif len(col) == 3:
                geom.set("rgba", self.as_str(spec["color"] + [1]))
            elif len(col) == 2:
                geom.set("rgba", f"{col[0]} {col[0]} {col[0]} {col[1]}")
            elif len(col) == 1:
                geom.set("rgba", f"{col[0]} {col[0]} {col[0]} 1.0")
            else:
                raise Exception("NIY")

        # has inertia
        if "mass" in spec:
            if geom is not None:
                geom.set("mass", str(spec["mass"]))
            else:
                i = ET.SubElement(
                    a, "inertial", {"pos": "0 0 0", "mass": str(spec["mass"]), "diaginertia": "1e-5 1e-5 1e-5"}
                )

        # recurse through all children (depth first)
        for ch in f.getChildren():
            spec = ch.asDict()
            isFree = "joint" in spec and spec["joint"] == "free"
            if not isFree:
                self.addFrame(ch, a)

    def dump(self):
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ", level=0)
        ET.dump(tree)
        tree.write("z.xml")

    def str(self):
        return ET.tostring(self.root)


M = MujocoWriter(C)
M.dump()
xml = M.str()
# C.view(True)
# exit()

xml2 = """
<mujoco>
<default class="viz">
<geom type="mesh"/>
</default>
<asset>
<mesh name="panda_link0_mesh" file="/home/mtoussai/.local/venv/lib/python3.12/site-packages/robotic/rai-robotModels/panda/meshes/link0.stl" />
</asset>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" mesh="panda_link0_mesh"/>
    <geom name="green_sphere" type="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

print(C.getJointNames())


class MjSim:
    tau_sim = 0.01

    def __init__(self, xml, view=True, Cinit=None):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # n = mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_QPOS)
        if Cinit is not None:
            self.set_state(Cinit)

        self.ctrl = ry.BSpline()
        self.ctrl.set(2, self.data.ctrl.reshape(1,-1), [0])
        self.ctrl_dim = self.data.ctrl.size

        self.sim_time = 0.

    def set_state(self, C):
        q = C.getJointState()
        self.data.qpos[:q.size] = q
        # mujoco.mj_setState(self.m,self.d,q, mujoco.mjtState.mjSTATE_QPOS)
        # mujoco.mj_setState(m,d,q, mujoco.mjtState.mjSTATE_CTRL)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def run(self, T, Csync=None, view=True):
        Tstop = self.sim_time + T
        qn = C.getJointDimension()
        while self.sim_time < Tstop:
            ref = self.ctrl.eval3(self.sim_time)
            self.data.ctrl = ref[0]
            self.step()
            if view:
                if self.viewer is not None:
                    self.viewer.sync()
                q = self.data.qpos[:qn]
                Csync.setJointState(q)
                C.view(False, f"time: {self.sim_time}")
                time.sleep(self.tau_sim)

            self.sim_time += self.tau_sim


sim = MjSim(xml, False, C)
for k in range(100):
    sim.run(.1, C, True)
    q = C.getJointState()
    q = q[:sim.ctrl_dim]
    q += .1 * np.random.randn(q.size)
    sim.ctrl.overwriteSmooth(q.reshape(1,-1), [.2], sim.sim_time)
