# pip install robotic
#  details: https://github.com/MarcToussaint/robotic

# download models:
# curl -o shapenetModels.zip https://tubcloud.tu-berlin.de/s/r96oXEboEeozyMw/download
# unzip shapenetModels.zip

import numpy as np
import robotic as ry

print(ry.compiled())
ry.params_add({
    'physx/defaultFriction': 3.,  #reduce slip
    })

def displayPcl(points, normals):
    C = ry.Config()
    C.addFrame('pcl') \
        .setPosition([0,0,1]) \
        .setPointCloud(points, [255,0,0])
    
    lines = np.concatenate((pts, pts + .02*normals), axis=1).reshape(-1,3)
    C.addFrame('normals') \
        .setPosition([0,0,1]) \
        .setLines(lines, [255,0,0])

    C.view(True, 'displayPcl method')

### low-level interface

SG = ry.DataGen.ShapenetGrasps()
SG.setOptions(verbose=2)

SG.loadObject(shape=3)

pts = SG.getPointCloud()
normals = SG.getPointNormals()
print('point cloud size: ', pts.shape)
displayPcl(pts, normals)

for i in range(5):
    pose = SG.sampleGraspPose()
    print('candidate pose: ', pose)

    SG.setGraspPose(pose)

    scores = SG.evaluateGrasp()
    poses = SG.getEvalGripperPoses()
    print('scores: ', scores)
    print('poses: (esp poses[1] is interesting, which is after finger close)\n', poses)


### batch data generation interface

SG = ry.DataGen.ShapenetGrasps()

SG.setOptions(startShape=3, endShape=4, verbose=1) #verbose=2 for slow, verbose=0 for silent

X, Z, S = SG.getSamples(5)

SG.displaySamples(X, Z, S)

for i in range(X.shape[0]):
    SG.resetObjectPose()
    scores = SG.evaluateSample(X[i], Z[i]).reshape(-1)
    #scores is a list of numbers for different tests, should all be positive
    print(f'sample {i}:\n  valid: {min(scores)>0.}\n  scores: {scores}\n  data: {S[i]}')