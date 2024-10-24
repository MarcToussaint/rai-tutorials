import robotic as ry
import numpy as np
import time
from typing import Union

class ManipulationModelling():

    def __init__(self, info=str()):
        """
        Initialize a new instance of the ManipulationModelling class.

        Args:
            info (str, optional): An optional string for providing additional information or description related to this
                                manipulation instance. Default is an empty string.
        """
        self.komo = None
        self.info = info

    def setup_inverse_kinematics(self, C: ry.Config, homing_scale: float = 1e-1, accumulated_collisions: bool = True, joint_limits: bool = True, quaternion_norms: bool = False):
        """
        Set up a single-phase inverse kinematics problem with optional constraints.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            homing_scale (float, optional): The weight for the homing control objective, which defines the cost of deviation from 
                                            the default (home) position. Default is 0.1.
            accumulated_collisions (bool, optional): If True, imposes a constraint on accumulated collisions to minimize
                                                     collisions between objects. Default is True.
            joint_limits (bool, optional): If True, imposes constraints on joint limits to ensure the robot's joints 
                                            stay within their allowable range. Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure stable 
                                            orientation representations. Default is False.

        Raises:
            AssertionError: If the KOMO problem is already initialized (self.komo is not None).
        """
        assert self.komo==None
        self.komo = ry.KOMO(C, 1., 1, 0, accumulated_collisions)
        self.komo.addControlObjective([], 0, homing_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_sequence(self, C: ry.Config, K: int, homing_scale: float = 1e-2, velocity_scale: float = 1e-1, accumulated_collisions: bool = True, joint_limits: bool = True, quaternion_norms: bool = False):
        """
        Sets up the KOMO problem to control a sequence of joint configurations, adding objectives for homing, velocity, and constraints 
        like collision avoidance, joint limits, and quaternion normalization.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            K (int): The number of phases (time steps) for the KOMO problem.
            homing_scale (float, optional): The weight for the homing control objective, which defines the cost of deviation from 
                                            the default (home) position. Default is 0.01.
            velocity_scale (float, optional): The weight for the velocity control objective, which penalizes excessive velocities in joint space.
                                    Default is 0.1.
            accumulated_collisions (bool): If True, adds an equality constraint to avoid accumulated collisions.
                                        Default is True.            
            joint_limits (bool): If True, adds an inequality constraint to enforce joint limits.
                                Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure stable 
                                            orientation representations. Default is False.
        Raises:
            AssertionError: If the KOMO problem is already initialized (self.komo is not None).
        """

        assert self.komo==None
        self.komo = ry.KOMO(C, K, 1, 1, accumulated_collisions)
        self.komo.addControlObjective([], 0, homing_scale)
        self.komo.addControlObjective([], 1, velocity_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_motion(self, C: ry.Config, K: int, steps_per_phase: int, homing_scale: float, acceleration_scale: float, accumulated_collisions: bool, joint_limits: bool, quaternion_norms: bool):
        """
        Sets up the KOMO problem for motion planning, adding objectives for homing, acceleration, and constraints like collision avoidance, 
        joint limits, quaternion normalization.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            K (int): The number of phases (time steps) for the KOMO problem.
            steps_per_phase (int): The number of time steps per phase in the motion sequence.
            homing_scale (float): The weight for the homing control objective, determining the cost of deviation from the default (home) position.
            acceleration_scale (float): The weight for the acceleration control objective, penalizing excessive accelerations in joint space.
            accumulated_collisions (bool): If True, adds an equality constraint to avoid accumulated collisions.
            joint_limits (bool): If True, adds an inequality constraint to enforce joint limits.
            quaternion_norms (bool): If True, imposes a quaternion normalization constraint to ensure valid orientation representations.
        
        Raises:
            AssertionError: If the KOMO problem is already initialized (self.komo is not None).

        Details:
            -   Ensuring zero velocity at the end of the motion.
        """
        assert self.komo==None
        self.komo = ry.KOMO(C, K, steps_per_phase, 2, accumulated_collisions)
        if homing_scale>0.:
            self.komo.addControlObjective([], 0, homing_scale)
        self.komo.addControlObjective([], 2, acceleration_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

        # zero vel at end
        self.komo.addObjective([float(K)], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)

    def setup_pick_and_place_waypoints(self, C, gripper, obj, homing_scale=1e-2, velocity_scale=1e-1, accumulated_collisions=True, joint_limits=True, quaternion_norms=False):
        """
        Set up a two-phase pick-and-place motion problem with a pick switch at timestep 1, and a place switch at timestep 2.

        Args:
            C (ry.Config): The current robotic configuration, representing the kinematic structure (a tree of frames essentially).
            gripper (str): The name of the gripper that will pick and place the object.
            obj (str): The name of the object to be picked and placed.
            homing_scale (float, optional): The weight for the homing control objective, which defines the cost of deviation from 
                                            the default (home) position. Default is 0.1.
            velocity_scale (float, optional): The weight for the velocity control objective, which penalizes excessive velocities in joint space.
                                    Default is 0.1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize
                                                     collisions between objects. Default is True.
            joint_limits (bool, optional): If True, imposes constraints on joint limits to ensure the robot's joints 
                                            stay within their allowable range. Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure stable 
                                            orientation representations. Default is False.

        Raises:
            AssertionError: If the KOMO problem is already initialized (self.komo is not None).

        Details:
            - The place mode switch at the final time might seem obselete, but this switch also implies the geometric constraints of placeOn

        """
        assert self.komo==None
        self.setup_sequence(C, 2, homing_scale, velocity_scale, accumulated_collisions, joint_limits, quaternion_norms)

        #-- option 1: old-style mode switches: //a temporary free stable joint gripper -> object
        #self.komo.addModeSwitch([1.,-1.], ry.SY.stable, [gripper, obj], True)
        #-- option 2: a permanent free stable gripper->grasp joint; and a snap grasp->object
        self.add_stable_frame(ry.JT.free, gripper, 'obj_grasp', initFrame=obj)
        self.snap_switch(1., 'obj_grasp', obj)
        #-- option 3: a permanent free stable object->grasp joint; and a snap gripper->grasp
        # self.add_stable_frame(ry.JT.free, obj, 'obj_grasp', initFrame=obj)
        # self.snap_switch(1., gripper, 'obj_grasp')

    def setup_point_to_point_motion(self, C, q1, homing_scale=1e-2, acceleration_scale=1e-1, accumulated_collisions=True, joint_limits=True, quaternion_norms=False):       
        """
        Set up a one-phase fine-grained motion problem with second-order (acceleration) control costs.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            q0 (list[float]): The initial configuration of the robot, represented as a list of joint values.
            q1 (list[float]): The target configuration of the robot, represented as a list of joint values.
            homing_scale (float, optional): The weight for the homing control objective, which defines the cost of deviation from 
                                            the default (home) position. Default is 0.1.
            acceleration_scale (float, optional): The scaling factor for the acceleration control objective, 
                                                influencing the robot's movement acceleration. Default is 0.1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize 
                                                    or avoid collisions between objects. Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure stable 
                                            orientation representations. Default is False.
        
        Raises:
            AssertionError: If the KOMO problem is already initialized (self.komo is not None).

        """
        assert self.komo==None
        self.setup_motion(C, 1, 32, homing_scale, acceleration_scale, accumulated_collisions, joint_limits, quaternion_norms)

        self.komo.initWithWaypoints([q1], 1, interpolate=True, qHomeInterpolate=.5, verbose=0)
        self.komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], target=q1)

    def setup_point_to_point_rrt(self, C: ry.Config, q0: list[float], q1: list[float], explicitCollisionPairs: list[tuple[str, str]]):
        """
        Sets up a point-to-point RRT (Rapidly-exploring Random Tree) motion planning problem.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            q0 (list[float]): The initial joint configuration.
            q1 (list[float]): The target joint configuration.
            explicitCollisionPairs (list[tuple[str, str]]): A list of explicit frame pairs to be checked for collisions during the RRT planning.
        """        
        rrt = ry.PathFinder()
        rrt.setProblem(C, q0, q1)
        if len(explicitCollisionPairs):
            rrt.setExplicitCollisionPairs(explicitCollisionPairs)

    def add_stable_frame(self, jointType: ry.JT, parent: str, name: str, initFrame: Union[str, None] = None, markerSize: float = -1.0):
        """
        Adds a stable frame to the robot configuration with an optional visual marker.

        Args:
            jointType (ry.JT): The type of joint connecting the new frame to the parent.
            parent (str): The name of the parent frame to which the new frame is attached.
            name (str): The name of the new frame.
            initFrame (Union[str, None], optional): The initial frame configuration, either as a frame name or None. If a string is provided, the function will resolve it to a frame.
            markerSize (float, optional): The size of the visual marker to attach to the new frame. Default is -1 (no marker).
        """
        if isinstance(initFrame, str):
            initFrame = self.komo.getConfig().getFrame(initFrame)
        f = self.komo.addStableFrame(name, parent, jointType, True, initFrame)
        if markerSize>0.:
            f.setShape(ry.ST.marker, [.2])
            f.setColor([1., 0., 1.])
        #f.joint.sampleSdv=1.
        #f.joint.setRandom(self.komo.timeSlices.d1, 0)

    def grasp_top_box(self, time, gripper, obj, grasp_direction='xz'):
        """
        Grasp a box using a top-centered grasp with the gripper's axes fully aligned with the object's axes.

        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the object (box) to be grasped.
            grasp_direction (str, optional): Specifies the grasp direction by aligning specific axes of the gripper 
                                            and the object. Default is 'xz'.
                                            Possible values:
                                            - 'xz': Aligns XY, XZ, and YZ axes.
                                            - 'yz': Aligns YY, XZ, and YZ axes.
                                            - 'xy': Aligns XY, XZ, and ZZ axes.
                                            - 'zy': Aligns XX, XZ, and ZZ axes.
                                            - 'yx': Aligns YY, YZ, and ZZ axes.
                                            - 'zx': Aligns YX, YZ, and ZZ axes.

        Raises:
            Exception: If an invalid grasp_direction is provided.
        """
        if grasp_direction == 'xz':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'yz':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'xy':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'zy':
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'yx':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'zx':
            align = [ry.FS.scalarProductYX, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        else:
            raise Exception('pickDirection not defined:', grasp_direction)

        # position: centered
        self.komo.addObjective([time], ry.FS.positionDiff, [gripper, obj], ry.OT.eq, [1e1])

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], align[0], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[2], [obj, gripper], ry.OT.eq, [1e0])


    def grasp_box(self, time: float, gripper: str, obj: str, palm: str, grasp_direction: str='x', margin: float=.02):
        """
        Perform a general grasp of a box by squeezing along the specified grasp axis (resulting in three possible grasps of a box),
        ensuring no collision with the palm. 
        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the object (box) to be grasped.
            palm (str): The name of the palm or part of the robot that should avoid collisions with the object.
            grasp_direction (str, optional): The axis along which the gripper should grasp the box. Default is 'x'.
                                            Possible values:
                                            - 'x': Grasp along the X-axis.
                                            - 'y': Grasp along the Y-axis.
                                            - 'z': Grasp along the Z-axis.
            margin (float, optional): The margin for no-collision constraints between the box and the palm. Default is 0.02.

        Raises:
            Exception: If an invalid grasp_direction is provided.

        Details:
            - The Angle of the grasp is decided by inequalities on the grasp plan.
            - The position and orientation objectives ensure that the gripper is centered on the box, and the grasp axis is orthogonal to the target plane.
            - The margin parameter is used to add tolerance to the no-collision constraints between the box and palm.

        """
        if grasp_direction == 'x':
            xLine = np.array([[1, 0, 0]])
            yzPlane = np.array([[0, 1, 0],[0, 0, 1]])
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ]
        elif grasp_direction == 'y':
            xLine = np.array([[0, 1, 0]])
            yzPlane = np.array([[1, 0, 0],[0, 0, 1]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ]
        elif grasp_direction == 'z':
            xLine = np.array([[0, 0, 1]])
            yzPlane = np.array([[1, 0, 0],[0, 1, 0]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXY]
        else:
            raise Exception('grasp_direction not defined:', grasp_direction)

        boxSize = self.komo.getConfig().getFrame(obj).getSize()[:3]

        # position: center in inner target plane X-specific
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xLine*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*1e1, .5*boxSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*(-1e1), -.5*boxSize+margin)

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], align[0], [gripper, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def grasp_cylinder(self, time: float, gripper: str, obj: str, palm: str, margin: float=.02):
        """
        Perform a grasp of a cylinder by squeezing normally to the cylinder's axis, ensuring no collision with the palm.
        
        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the cylindrical object to be grasped.
            palm (str): The name of the palm or part of the robot that should avoid collisions with the object.
            margin (float, optional): The margin for no-collision constraints between the cylinder and the palm. Default is 0.02.

        Details:
            - Inequality constraint along the z-axis for positioning.

        """
        size = self.komo.getConfig().getFrame(obj).getSize()[:2]

        # position: center along axis, stay within z-range
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, np.array([[1, 0, 0],[0, 1, 0]])*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0, 0, 1]])*1e1, np.array([0.,0.,.5*size[0]-margin]))
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0, 0, 1]])*(-1e1), np.array([0.,0.,-.5*size[0]+margin]))

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], ry.FS.scalarProductXZ, [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def place_box(self, time: float, obj: str, table: str, palm: str, place_direction: str='z', margin: float=.02):
        """
        Placement of one box or cylinder onto another box (named table) in a specific direction

        Args:
            time (float): The time at which the placement is executed.
            obj (str): The name of the object (box or cylinder) to be placed.
            table (str): The name of the surface (table or another box) where the box will be placed.
            palm (str): The name of the palm or part of the robot that should avoid collisions during placement.
            place_direction (str, optional): The axis along which the box is placed on the surface. Default is 'z'.
                                            Possible values:
                                            - 'x': Place along the X-axis.
                                            - 'y': Place along the Y-axis.
                                            - 'z': Place along the Z-axis (default).
                                            - 'xNeg': Place along the negative X-axis.
                                            - 'yNeg': Place along the negative Y-axis.
                                            - 'zNeg': Place along the negative Z-axis.
            margin (float, optional): The margin to avoid collisions between the box and the table or other objects. Default is 0.02.
        
        Raises:
            Exception: If an invalid shape type for placing is provided.
        """
        zVectorTarget = np.array([0.,0.,1.])
        obj_frame = self.komo.getConfig().getFrame(obj)
        boxSize = obj_frame.getSize()
        if obj_frame.getShapeType()==ry.ST.ssBox:
            boxSize = boxSize[:3]
        elif obj_frame.getType()==ry.ST.ssCylinder:
            boxSize = [boxSize[1], boxSize[1], boxSize[0]] 
        else:
            raise Exception('NIY')

        tableSize = self.komo.getConfig().getFrame(table).getSize()[:3]
        if place_direction == 'x':
            relPos = .5*(boxSize[0]+tableSize[2])
            zVector = ry.FS.vectorX
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductYX]
        elif place_direction == 'y':
            relPos = .5*(boxSize[1]+tableSize[2])
            zVector = ry.FS.vectorY
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductYY]
        elif place_direction == 'z':
            relPos = .5*(boxSize[2]+tableSize[2])
            zVector = ry.FS.vectorZ
            align = [ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif place_direction == 'xNeg':
            relPos = .5*(boxSize[0]+tableSize[2])
            zVector = ry.FS.vectorX
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductYX]
        elif place_direction == 'yNeg':
            relPos = .5*(boxSize[1]+tableSize[2])
            zVector = ry.FS.vectorY
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductYY]
        elif place_direction == 'zNeg':
            relPos = .5*(boxSize[2]+tableSize[2])
            zVector = ry.FS.vectorZ
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        else:
            raise Exception('place_direction not defined:', place_direction)

        # position: above table, inside table
        self.komo.addObjective([time], ry.FS.positionDiff, [obj, table], ry.OT.eq, 1e1*np.array([[0, 0, 1]]), np.array([.0, .0, relPos]))
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, 1e1*np.array([[1, 0, 0],[0, 1, 0]]), .5*tableSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, -1e1*np.array([[1, 0, 0],[0, 1, 0]]), -.5*tableSize+margin)

        # orientation: Z-up
        self.komo.addObjective([time-.2, time], zVector, [obj], ry.OT.eq, [0.5], zVectorTarget)
        self.komo.addObjective([time-.2,time], align[0], [table, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [table, obj], ry.OT.eq, [1e0])

        # no collision with palm
        if palm != None:
           self.komo.addObjective([time-.3, time], ry.FS.distance, [palm, table], ry.OT.ineq, [1e1], [-.001])

    def straight_push(self, time_interval: list[float], obj: str, gripper: str, table: str):
        """
        Define a straight push motion for the gripper to push an object across a table.

        Args:
            times (list[float]): A list of two time points specifying the start and end times for the pushing motion.
            obj (str): The name of the object being pushed.
            gripper (str): The name of the gripper that will perform the push.
            table (str): The name of the table where the push occurs.

        Behavior:
            - Adds two helper frames ('_push_start' and '_push_end') attached to the table and object to define the
            start and end points of the pushing motion.
            - Ensures the start and end frames are aligned in both orientation and position, imposing constraints on
            their alignment and ensuring a minimum distance between them.
            - Ensures the gripper is in contact with the object and aligns it with the start position at the beginning
            of the motion.
            - The object is constrained to follow a straight path and maintain its orientation at the end of the push.
        """ 
        #start & end helper frames
        helperStart = f'_straight_pushStart_{gripper}_{obj}_{time_interval[0]}'
        helperEnd = f'_straight_pushEnd_{gripper}_{obj}_{time_interval[1]}'
        if not self.komo.getConfig().getFrame(helperStart, False):
            self.add_stable_frame(ry.JT.hingeZ, table, helperStart, obj, .3)
        if not self.komo.getConfig().getFrame(helperEnd, False):
            self.add_stable_frame(ry.JT.transXYPhi, table, helperEnd, obj, .3)

        #-- couple both frames symmetricaly
        #aligned orientation
        self.komo.addObjective([time_interval[0]], ry.FS.vectorYDiff, [helperStart, helperEnd], ry.OT.eq, [1e1])
        #aligned position
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [helperEnd, helperStart], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [helperStart, helperEnd], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        #at least 2cm appart, positivenot !not  direction
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [helperEnd, helperStart], ry.OT.ineq, -1e2*np.array([[0., 1., 0.]]), [.0, .02, .0])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [helperStart, helperEnd], ry.OT.ineq, 1e2*np.array([[0., 1., 0.]]), [.0, -.02, .0])

        #gripper touch
        self.komo.addObjective([time_interval[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.02])
        #gripper start position
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.ineq, 1e1*np.array([[0., 1., 0.]]), [.0, -.02, .0])
        #gripper start orientation
        self.komo.addObjective([time_interval[0]], ry.FS.scalarProductYY, [gripper, helperStart], ry.OT.ineq, [-1e1], [.2])
        self.komo.addObjective([time_interval[0]], ry.FS.scalarProductYZ, [gripper, helperStart], ry.OT.ineq, [-1e1], [.2])
        self.komo.addObjective([time_interval[0]], ry.FS.vectorXDiff, [gripper, helperStart], ry.OT.eq, [1e1])

        #obj end position
        self.komo.addObjective([time_interval[1]], ry.FS.positionDiff, [obj, helperEnd], ry.OT.eq, [1e1])
        #obj end orientation: unchanged
        self.komo.addObjective([time_interval[1]], ry.FS.quaternion, [obj], ry.OT.eq, [1e1], [], 1); #qobjPose.rot.getArr4d())
    
        return helperStart

    def pull(self, times: list[float], obj: str, gripper: str, table: str):
        """
        Define a pulling motion where the gripper pulls an object along the table surface while maintaining a fixed downward orientation.

        Args:
            times (list[float]): A list of two time points specifying the start and end times for the pulling motion.
            obj (str): The name of the object being pulled.
            gripper (str): The name of the gripper performing the pull.
            table (str): The name of the table or surface on which the object is being pulled.

        """
        self.add_stable_frame(ry.JT.transXYPhi, table, '_pull_end', obj)
        self.komo.addObjective([times[0]], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[1]], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[0]], ry.FS.vectorZ, [obj], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[1]], ry.FS.vectorZ, [obj], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[1]], ry.FS.positionDiff, [obj, '_pull_end'], ry.OT.eq, [1e1])
        self.komo.addObjective([times[0]], ry.FS.positionRel, [gripper, obj], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 1., 0.]]), np.array([0, 0, 0]))
        self.komo.addObjective([times[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.005])

    def no_collisions(self, time_interval: list[float], objs: list[str], margin: float = 0.001):
        """
        Add inequality constraints on the distance between multiple objects to ensure no collisions between
        multiple objects over a specified time interval.

        Args:
            time_interval (list[float]): A list containing two elements that specify the start and end times 
                                        for which the negDistance constraints are applicable.
            objs (list[str]): A list of object names for which collision avoidance is to be ensured.
            margin (float, optional): The minimum required distance between objects to prevent collisions. 
                                    Default value is 0.001 meters.
        """

        while len(objs) > 1:
            comp = objs[0]
            del objs[0]
            for obj in objs:
                self.komo.addObjective(time_interval, ry.FS.negDistance, [comp, obj], ry.OT.ineq, [1e1], [-margin])

    def snap_switch(self, time, parent, obj):
        '''
        a kinematic mode switch, where at given time the obj becomes attached to parent with zero relative transform
        the parent is typically a stable_frame (i.e. a frame that has parameterized but stable (i.e. constant) relative transform)
        '''
        self.komo.addRigidSwitch(time, [parent, obj])

    def switch_place():
        '''
        a kinematic mode switch, where obj becomes attached to table, with a 3D parameterized (XYPhi) stable relative pose
        this requires obj and table to be boxes and assumes default placement alone z-axis
        more general placements have to be modelled with switch_pick (table picking the object) and additinal user-defined geometric constraints
        '''

    def target_position():
        '''
        impose a specific 3D target position on some object
        '''

    def target_relative_xy_position(self, time: float, obj: str, relativeTo: str, pos: list[float]):
        """
        Impose a specific 3D target position on an object relative to another frame at a given time.

        Args:
            time (float): The time at which the position constraint is applied.
            obj (str): The name of the object whose position is being constrained.
            relativeTo (str): The name of the reference frame relative to which the object's position is defined.
            pos (list[float]): A list of two or three floats representing the target position relative to `relativeTo`.
                            If only two values are provided, the z-component is set to 0.
        """
        if len(pos)==2:
            pos.append(0.)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, relativeTo], ry.OT.eq, scale=1e1*np.array([[1,0,0],[0,1,0]]), target=pos)

    def target_x_orientation(self, time: float, obj: str, x_vector: list[float]):
        """
        Align the x-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's x-axis.
                                    This vector defines the target direction for the object's x-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorX, [obj], ry.OT.eq, scale=[1e1], target=x_vector)

    def target_y_orientation(self, time: float, obj: str, y_vector: list[float]):
        """
        Align the y-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's y-axis.
                                    This vector defines the target direction for the object's y-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorY, [obj], ry.OT.eq, scale=[1e1], target=y_vector)

    def target_z_orientation(self, time: float, obj: str, z_vector: list[float]):
        """
        Align the z-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's z-axis.
                                    This vector defines the target direction for the object's z-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorZ, [obj], ry.OT.eq, scale=[1e1], target=z_vector)

    def bias(self, time: float, qBias: list[float], scale: float = 1.0):
        """
        Impose a square potential bias directly in joint space.

        Args:
            time (float): The time at which to impose the bias.
            qBias (list[float]): A list of target joint angles or positions. This represents the desired configuration for the robot's joints.
            scale (float, optional): The scaling factor for the bias. This controls the strength of the imposed bias.
                                    Default value is 1.0.

        Details:
            - This method adds an Sum-of-squares objective to the optimization problem to steer the system towards the specified joint configuration `qBias`.
        """
        self.komo.addObjective([time], ry.FS.qItself, [], ry.OT.sos, scale=scale, target=qBias)

    def retract(self, time_interval: list[float], gripper: str, dist: float = 0.03):
        """
        Define a retract motion for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the retract motion.
            gripper (str): The name of the gripper that will perform the retract motion.
            dist (float, optional): The distance to retract relative to the grippers z-axis. 
                                    Default value is 0.03 meters.
        """       
        helper = f'_{gripper}_retract_{time_interval[0]}'
        f = self.komo.getFrame(gripper, time_interval[0])
        self.add_stable_frame(ry.JT.none, '', helper, f)
        #  self.komo.view(True, helper)

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), target = [0., 0., dist])

    def approach(self, time_interval: list[float], gripper: str, dist: float = 0.03):
        """
        Define an approach motion for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the approach motion.
            gripper (str): The name of the gripper that will perform the approach motion.
            dist (float, optional): The distance to approach relative to the gripper's z-axis. 
                                    Default value is 0.03 meters.
        """
        helper = f'_{gripper}_approach_{time_interval[1]}'
        f = self.komo.getFrame(gripper, time_interval[1])
        self.add_stable_frame(ry.JT.none, '', helper, f)
        #  self.komo.view(True, helper)

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), target = [0., 0., dist])

    def retractPush(self, time_interval: list[float], gripper: str, dist: float):
        """
        Define a retract motion with a push for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the retract motion.
            gripper (str): The name of the gripper that will perform the retract push motion.
            dist (float): The distance to retract relative to the gripper's z-axis.
        """       
        helper = f'_{gripper}_retractPush_{time_interval[0]}'
        f = self.komo.getFrame(gripper, time_interval[0])
        self.add_stable_frame(ry.JT.none, '', helper, f)
        #  self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1,3},{1,0,0]]))
        #  self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1, 0, 0]]))
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, * np.array([[0, 1, 0]]), [0., -dist, 0.])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), [0., 0., dist])

    def approachPush(self, time_interval: list[float], gripper: str, dist: float):
        """
        Define an approach motion with a push for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the approach motion.
            gripper (str): The name of the gripper that will perform the approach push motion.
            dist (float): The distance to approach relative to the gripper's z-axis.
        """    
        helper = f'_{gripper}_approachPush_{time_interval[1]}'
        f = self.komo.getFrame(gripper, time_interval[1])
        self.add_stable_frame(ry.JT.none, '', helper, f)
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1, 0, 0]]))
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, * np.array([[0, 1, 0]]), [0., -dist, 0.])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), [0., 0., dist])
        
    def solve(self, verbose: int=1) -> list[list[float]]:
        """
        Finding a feasible path or trajectory by solving a nonlinear optimization problem formulated in KOMO, or using RRT, respectively.

        Args:
            verbose (int, optional): Sets the verbosity level for logging and visualization.
                - 0: No output.
                - 1: Minimal output, showing feasibility of the solution.
                - 2: Detailed output, including solver information and failure reports.
                - 3: Full output with real-time playback of the trajectory. Default is 1.
            
        Returns:
            list[list[float]]: The computed path or trajectory as a list of 7D joint angles, if a solution is found.
                            Returns `None` if the optimization fails or no problem is defined.
        
        Raises:
            Exception: If neither KOMO nor RRT is defined for solving the problem.
        """
        if self.komo:
            sol = ry.NLP_Solver()
            sol.setProblem(self.komo.nlp())
            sol.setOptions(damping=1e-1, verbose=verbose-1, stopTolerance=1e-3, maxLambda=100., stopInners=20, stopEvals=200)
            self.ret = sol.solve()
            if self.ret.feasible:
                self.path = self.komo.getPath()
            else:
                self.path = None
            if verbose>0:
                if not self.ret.feasible:
                    print(f'  -- infeasible:{self.info}\n     {self.ret}')
                    if verbose>1:
                        print(self.komo.report(False, True))
                        self.komo.view(True, f'failed: {self.info}\n{self.ret}')
                    if verbose>2:
                        while(self.komo.view_play(True, 1.)):
                            pass
                else:
                    print(f'  -- feasible:{self.info}\n     {self.ret}')
                    if verbose>2:
                        self.komo.view(True, f'success: {self.info}\n{self.ret}')
                    if verbose>3:
                        while(self.komo.view_play(True, 1.)):
                            pass

        elif self.rrt:
            ret = self.rrt.solve()
            if ret.feasible:
                self.path = ret.x
            else:
                self.path = None

        else:
            raise Exception('no problem defined')
            
        return self.path

    def debug(self, listObjectives, plotOverTime):
    #     cout <<'  -- DEBUG: ' <<info <<endl
    #     cout <<'  == solver return: ' <<*ret <<endl
    #     cout <<'  == all KOMO objectives with increasing errors:\n' <<self.komo.report(False, listObjectives, plotOverTime) <<endl
    # #  cout <<'  == objectives sorted by error and Lagrange gradient:\n' <<sol.reportLagrangeGradients(self.komo.featureNames) <<endl
    #     cout <<'  == view objective errors over slices in gnuplot' <<endl
    #     cout <<'  == scroll through solution in display window using SHIFT-scroll' <<endl
        self.komo.view(True, f'debug: {info}\n{self.ret}')

    def play(self, C: ry.Config, duration: float = 1.):
        """
        Play back a trajectory by setting joint states at each step.

        Args:
            C (ry.Config): The current robot configuration, representing the kinematic structure as a tree of frames.
            duration (float, optional): The total duration for playing back the trajectory.
                                        The default is 1 second.
        """
        for t in range(self.path.shape[0]):
            C.setJointState(self.path[t])
            C.view(False, f'step {t}\n{self.info}')
            time.sleep(duration/self.path.shape[0])

    def sub_motion(self, phase, fixEnd=True, homing_scale=1e-2, acceleration_scale=1e-1, accumulated_collisions=True, quaternion_norms=False) -> 'ManipulationModelling':
        """
        Create a sub-motion plan for a specific phase using KOMO and return a ManipulationModelling instance.

        Args:
            phase (int): The phase number for which the sub-motion is to be planned.
            fixEnd (bool, optional): If True, ensures the final configuration (q1) is fixed for the motion. Default is True.
            homing_scale (float, optional): The weight for the homing control objective, which defines the cost of deviation from 
                                            the default (home) position. Default is 0.1.
            acceleration_scale (float, optional): The scaling factor for the acceleration minimization objective. Default is 1e-1.
            accumulated_collisions (bool, optional): If True, enables accumulated collision constraints during the sub-motion planning. Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure stable 
                                            orientation representations. Default is False.
        Returns:
            ManipulationModelling: A new instance of the ManipulationModelling class, configured for the sub-motion plan of the given phase.
        """
        (C, q0, q1) = self.komo.getSubProblem(phase)
        manip = ManipulationModelling(f'sub_motion_{phase}--{self.info}')
        manip.setup_point_to_point_motion(C, q1, homing_scale, acceleration_scale, accumulated_collisions, quaternion_norms)
        return manip


    def sub_rrt(self, phase: int, explicitCollisionPairs: list[str]=[]) -> 'ManipulationModelling':
        """
        Create a sub-motion plan for a specific phase using RRT and return a ManipulationModelling instance.

        Args:
            phase (int): The phase number for which the sub-motion is to be planned.
            explicitCollisionPairs (list[str], optional): A list of object pairs for which explicit collision avoidance should be enforced. Default is an empty list.

        Returns:
            ManipulationModelling: A new instance of the ManipulationModelling class, configured for the sub-motion plan of the given phase.
        """
        (C, q0, q1) = self.komo.getSubProblem(phase)
        manip = ManipulationModelling(f'sub_rrt_{phase}--{self.info}')
        manip.setup_point_to_point_rrt(C, q0, q1, explicitCollisionPairs)
        return manip
    
    @property
    def feasible(self):
        return self.ret.feasible
    
