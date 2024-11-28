# pip install robotic
#  details: https://github.com/MarcToussaint/robotic

# download models:
# curl -o shapenetModels.zip https://tubcloud.tu-berlin.de/s/r96oXEboEeozyMw/download
# unzip shapenetModels.zip

import numpy as np
import robotic as ry

print(ry.compiled())


### low-level interface

SG = ry.DataGen.ShapenetGrasps()
SG.setOptions(verbose=2)

SG.loadObject(shape=3)
pcl = SG.getPointCloud()
print('point cloud size: ', pcl.shape)

pose = SG.sampleGraspPose()
print('candidate pose: ', pose)

scores = SG.evaluateGrasp()
print('scores: ', scores)


### batch data generation interface

SG = ry.DataGen.ShapenetGrasps()

SG.setOptions(startShape=3, numShapes=1, verbose=1) #verbose=2 for slow, verbose=0 for silent
SG.setPhysxOptions(motorKp=20000., motorKd=500., angularDamping=1., defaultFriction=3.)

X, Z, S = SG.getSamples(20)

SG.displaySamples(X, Z, S)

for i in range(X.shape[0]):
    scores = SG.evaluateSample(X[i], Z[i]).reshape(-1)
    #scores is a list of numbers for different tests, should all be positive
    print(f'sample {i}:\n  valid: {min(scores)>0.}\n  scores: {scores}\n  data: {S[i]}')