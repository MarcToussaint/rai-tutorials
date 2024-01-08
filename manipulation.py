import robotic as ry
import random
import numpy as np

class ManipulationModelling():

    def __init__(self, C, komo=None):
        self.C = C
        self.komo = komo

    def setup_inverse_kinematics(self, homing_scale=1e-1, accumulated_collisions=True, quaternion_norms=False):
        """
        setup a 1 phase single step problem
        """
        self.komo = ry.KOMO(self.C, 1., 1, 0, accumulated_collisions)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        if quaternion_norms:
            self.komo.addQuaternionNorms()
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])
        self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])


    def setup_pick_and_place_waypoints(self, gripper, obj, homing_scale=1e-2, velocity_scale=1e-1, accumulated_collisions=True, quaternion_norms=False):
        """
        setup a 2 phase pick-and-place problem, with a pick switch at time 1, and a place switch at time 2
        the place mode switch at the final time two might seem obselete, but this switch also implies the geometric constraints of placeOn
        """
        self.komo = ry.KOMO(self.C, 2., 1, 1, accumulated_collisions)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        self.komo.addControlObjective([], order=1, scale=velocity_scale)
        if quaternion_norms:
            self.komo.addQuaternionNorms()
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])
        self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])

        self.komo.addModeSwitch([1.,-1.], ry.SY.stable, [gripper, obj], True)

    def setup_point_to_point_motion(self, q0, q1, homing_scale=1e-2, acceleration_scale=1e-1, accumulated_collisions=True, quaternion_norms=False, helpers=[]):
        """
        setup a 1 phase fine-grained motion problem with 2nd order (acceleration) control costs
        """
        self.C.setJointState(q1)
        for frame in helpers:
            f = self.C.getFrame(f'_{frame}_end', False)
            if not f:
                f = self.C.addFrame(f'_{frame}_end')
            f_org = self.C.frame(frame)
            f.setPosition(f_org.getPosition())
            f.setQuaternion(f_org.getQuaternion())
            #f.setShape(ry.ST.marker, [.1])
            #f.setColor([0,0,1])
        self.C.setJointState(q0)
        for frame in helpers:
            f = self.C.getFrame(f'_{frame}_start', False)
            if not f:
                f = self.C.addFrame(f'_{frame}_start')
            f_org = self.C.frame(frame)
            f.setPosition(f_org.getPosition())
            f.setQuaternion(f_org.getQuaternion())
            #f.setShape(ry.ST.marker, [.1])
            #f.setColor([0,1,0])
            
        self.komo = ry.KOMO(self.C, 1., 32, 2, False)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        self.komo.addControlObjective([], order=2, scale=acceleration_scale)
        if quaternion_norms:
            self.komo.addQuaternionNorms()

        # zero vel at end
        self.komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], order=1);

        # end point
        self.komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], target=q1);

        


    def grasp_top_box(self, time, gripper, obj, grasp_direction='xz'):
        """
        grasp a box with a centered top grasp (axes fully aligned)
        """
        if grasp_direction == 'zx':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'zy':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'yx':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'yz':
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'xy':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'xz':
            align = [ry.FS.scalarProductYX, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        else:
            raise Exception('pickDirection not defined:', grasp_direction)

        # position: centered
        self.komo.addObjective([time], ry.FS.positionDiff, [gripper, obj], ry.OT.eq, [1e1])

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], align[0], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[2], [obj, gripper], ry.OT.eq, [1e0])


    def grasp_box(self, time, gripper, obj, palm, grasp_direction='x', margin=.02):
        """
        general grasp of a box, squeezing along provided grasp_axis (-> 3
        possible grasps of a box), where and angle of grasp is decided by
        inequalities on grasp plan and no-collision of box and palm
        """
        if grasp_direction == 'x':
            xLine = np.array([[1,0,0]])
            yzPlane = np.array([[0,1,0],[0,0,1]])
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ]
        elif grasp_direction == 'y':
            xLine = np.array([[0,1,0]])
            yzPlane = np.array([[1,0,0],[0,0,1]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ]
        elif grasp_direction == 'z':
            xLine = np.array([[0,0,1]])
            yzPlane = np.array([[1,0,0],[0,1,0]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXY]
        else:
            raise Exception('pickDirection not defined:', pickDirection)

        boxSize = self.C.frame(obj).getSize()[:3]

        # position: center in inner target plane X-specific
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xLine*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*1e1, .5*boxSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*(-1e1), -.5*boxSize+margin)

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], align[0], [gripper, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def grasp_cylinder(self, time, gripper, obj, palm, margin=.02):
        """
        general grasp of a cylinder, with squeezing the axis normally,
        inequality along z-axis for positioning, and no-collision with palm
        """
        size = self.C.frame(obj).getSize()[:2]

        # position: center along axis, stay within z-range
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, np.array([[1,0,0],[0,1,0]])*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0,0,1]])*1e1, [0.,0.,.5*size[0]-margin])
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0,0,1]])*(-1e1), [0.,0.,-.5*size[0]+margin])

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], ry.FS.scalarProductXZ, [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def place_box(self, time, obj, table, palm, place_direction='z', margin=.02):
        """
        placement of one box on another
        """
        zVectorTarget = np.array([0.,0.,1.])
        boxSize = self.C.frame(obj).getSize()[:3]
        tableSize = self.C.frame(table).getSize()[:3]
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

        # position: above table, inside table
        self.komo.addObjective([time], ry.FS.positionDiff, [obj, table], ry.OT.eq, 1e1*np.array([[0,0,1]]), [.0, .0, relPos])
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, 1e1*np.array([[1,0,0],[0,1,0]]), .5*tableSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, -1e1*np.array([[1,0,0],[0,1,0]]), -.5*tableSize+margin)

        # orientation: Z-up
        self.komo.addObjective([time-.2, time], zVector, [obj], ry.OT.eq, [0.5], zVectorTarget)
        self.komo.addObjective([time-.2,time], align[0], [table, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [table, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, table], ry.OT.ineq, [1e1], [-.001])

    def no_collision(self, time_interval, obj1, obj2, margin=.001):
        """
        inequality on distance between two objects
        """
        self.komo.addObjective(time_interval, ry.FS.negDistance, [obj1, obj2], ry.OT.ineq, [1e1], [-margin])

    def switch_pick():
        """
        a kinematic mode switch, where obj becomes attached to gripper, with freely parameterized but stable (=constant) relative pose
        """
    def switch_place():
        """
        a kinematic mode switch, where obj becomes attached to table, with a 3D parameterized (XYPhi) stable relative pose
        this requires obj and table to be boxes and assumes default placement alone z-axis
        more general placements have to be modelled with switch_pick (table picking the object) and additinal user-defined geometric constraints
        """
    def target_position():
        """
        impose a specific 3D target position on some object
        """
    def target_relative_xy_position(self, time, obj, relativeTo, pos):
        """
        impose a specific 3D target position on some object
        """
        if len(pos)==2:
            pos.append(0.)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, relativeTo], ry.OT.eq, scale=1e1*np.array([[1,0,0],[0,1,0]]), target=pos)

    def bias(self, time, qBias, scale=1e0):
        """
        impose a square potential bias directly in joint space
        """
        self.komo.addObjective([time], ry.FS.qItself, [], ry.OT.sos, target=qBias, scale=scale)

    def endeff_forward_motion(self, time_interval, gripper, q):
        self.C.setJointState(q)
        ori = self.C.frame(gripper).getRotationMatrix()
        x = ori[0:1]
        yz = ori[1:3]
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, 1e2*x, order=1)
        self.komo.addObjective(time_interval, ry.FS.angularVel, [gripper], ry.OT.eq, 1e1*yz, order=1)

    def endeff_retract(self, time_interval, gripper, dist=.05):
        helper = f'_{gripper}_start'
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1,0,0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0,0,1]]), target = [0., 0., dist])

    def endeff_approach(self, time_interval, gripper, dist=.05):
        helper = f'_{gripper}_end'
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1,0,0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0,0,1]]), target = [0., 0., dist])
        
    def solve(self, verbose=0):
        sol = ry.NLP_Solver()
        sol.setProblem(self.komo.nlp())
        sol.setOptions(damping=1e-3, verbose=verbose, stopTolerance=1e-3, maxLambda=100., stopEvals=200)
        ret = sol.solve()
        if verbose>0:
            print('solver return:', ret)
        if ret.feasible:
            return self.komo.getPath(), ret
        return None, ret


