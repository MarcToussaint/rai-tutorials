world {}

#thick radius floor and walls

floor (world){ shape:ssBox, Q:[0 0 -.05], size:[4.1 4.1 .1 .04], contact: 1 friction:10}

wall_right (world){ shape:ssBox, Q:[0 -2. 0.6], size:[4.1 .1 1.2 .04], contact: 1 }
wall_back (world){ shape:ssBox, Q:[2. 0 0.6], size:[.1 4.1 1.2 .04], contact: 1 }
wall_left (world){ shape:ssBox, Q:[0 2. 0.6], size:[4.1 .1 1.2 .04], contact: 1 }
wall_front (world){ shape:ssBox, Q:[-2. 0 0.6], size:[.1 4.1 1.2 .04] , contact: 1 }


base { X:[0 0 .1], multibody, multibody_gravity: false, mass:.1 }
jointX(base){ joint:transX, mass:.01, inertia:[.01,.01,.01], limits: [-2.,2.], motorLambda: .01, motorMass: .1, sampleUniform: 1. }
jointY(jointX){ joint:transY, mass:.01, inertia:[.01,.01,.01], limits: [-2.,2.], motorLambda: .01, motorMass: .1, sampleUniform: 1. }
ego(jointY) {
    shape:ssCylinder, size:[.2 .2 .02], logical,  mass: .1, contact: 1
}

Edit jointX { q: -1.  }
Edit jointY { q: -1.5 }

# transparent shapes are not at all instantiated in bullet! no physical interaction with regions
nogo (world){ shape:ssBox, Q:[0 0 0], size:[4 2 .025 .01], color:[0 0 0 .9] }
goal (world){ shape:ssBox, Q:[1 -1.5 0], size:[2.0 1.0 .025 .01], color:[1. .3 .3 .9] }

wall1 (world){ shape:ssBox, Q:[0 -1 0.3], size:[0.1 2 0.6 .04], contact: 1 }

obj {
  shape:ssBox, pose:[-1.4 -1.2 0.151], size:[.3 .3 0.3 .02], logical
  mass:1, friction:.1, contact: 1
}

obj2 {
  shape:ssBox, pose:[-.1 0.3 0.101], size:[1.5 0.2 .2 .02], logical
  mass:1, friction:.1, contact: 1
}

obj3 {
  shape:ssBox, pose:[-1 -0.8 0.101], size:[.5 0.8 .2 .02], logical
  mass:1, friction:.1, contact: 1
}

#to include arrows
#picture(nogo){
#    shape:quad, Q:[0 0 .1] size:[2, 1], texture='../res/arrowsup200x100.ppm', color:[1. 1. 1. .5] }

camera_init (world) {
    pose: [0, 0, 10, 0, 0, 1, 0]
}
