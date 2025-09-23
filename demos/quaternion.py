# READ THIS: https://www.user.tu-berlin.de/mtoussai/notes/quaternions.pdf
 
from robotic import Quaternion
import math
import numpy as np

q = Quaternion()

print('\n=== using set(...).normalize()):', q.set([1,1,0,0]).normalize())
print('    direct access:', q.w, q.x, q.y, q.z)
print('    as numpy array:', q.asArr())

## converting a quaternion to another representations
print('\n=== converting a quaternion')
print('    original: 90 degree rotation about x:', q)
print('    total rotation angle/pi:', q.getRad()/math.pi)
print('    roll-pitch-yaw/pi:', q.getRollPitchYaw()/math.pi)
print('    log (which is also the "rotation vector"):', q.getLog(), '  length/pi:', np.linalg.norm(q.getLog())/math.pi)
print('    matrix:', q.getMatrix())

## setting a quaternion in various ways:
q = Quaternion()
print('\n=== setting a quaternion')
print('   non-initialized!:', q)
print('   initialize "zero" (which here means identity!)', q.setZero())
print('   the minimal rotation to rotate vector a=(1,0,0) into vector b=(0,1,0)', q.setDiff([1,0,0], [0,1,0]))
print('   90 degrees about the [0,1,0] axis:', q.setRad(math.pi/2, [0,1,0]))
print('   random:', q.setRandom())
print('   exp(log(q)):', q.setExp(q.getLog()))
print('   quat(matrix(q)):', q.setMatrix(q.getMatrix()))
print('   quat(rollPitchWay(q)):', q.setRollPitchYaw(q.getRollPitchYaw()))

## geodesic interpolation
print('\n=== interpolation')
q0 = Quaternion().set([1,1,0,0]).normalize()
q1 = Quaternion().set([1,0,0,1]).normalize()
for t in np.arange(0, 1.1, 0.1):
    q_t = Quaternion().setInterpolateProper(t, q0, q1)
    q_t2 = q0 * Quaternion().setExp(t * ((-q0)*q1).getLog()) #same, but harder to remember
    print(f'   t:{t:.2},  q(t): {q_t},  via exp(log): {q_t2}')

## inverse
print('\n=== multiplication & inverse')
q.setRandom()
print('q:     ', q)
print('q*(-q):', q*(-q))
print('(-q)*q:', (-q)*q)
