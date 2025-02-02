#include <Kin/kin.h>
#include <Kin/frame.h>
#include <Kin/feature.h>
#include <Kin/simulation.h>
#include <Kin/viewer.h>
#include <Kin/F_geometrics.h>
#include <Kin/kin_physx.h>

#include <Geo/depth2PointCloud.h>

#include <iomanip>

//===========================================================================

void testComplexObjects(){
  StringA files = fromFile<StringA>(rai::raiPath("../rai-robotModels/shapenet/models/files"));

  rai::Configuration C;

  for(uint k=0;k<10;k++){
    str file = rai::raiPath("../rai-robotModels/shapenet/models/") + files(3+rnd(10));
    rai::Frame *obj = C.addH5Object(STRING("obj"<<k), file, 1);
    obj->set_X()->setRandom();
    obj->set_X()->pos.z += 1.;
  }

  rai::Simulation S(C, S._physx, 2);

  double tau=.01;
  Metronome tic(tau);

  for(uint t=0;t<4./tau;t++){
    tic.waitForTic();

    S.step({}, tau, S._none);
  }

  C.view(true);
}

//===========================================================================

void testPassive(const char* filename){
  rai::Configuration C;
  C.addFile(filename);
//  C.optimizeTree(true);

  rai::Simulation S(C, S._physx, 3);
//  rai::wait();

  double tau=.001;
  Metronome tic(1.*tau);

  for(double t=0.;t<4.;t+=tau){
    tic.waitForTic();
//    rai::wait(1.);

    S.step({}, tau, S._none);

    arr V;
    S.getState(NoArr, NoArr, V, NoArr);
    cout <<S.get_q() <<' ' <<S.get_qDot() <<' ' <<S.get_frameVelocities()[-1] <<V[-1] <<endl;

//    C.view(true, STRING("time:" <<t));
  }
}

//===========================================================================

void testSplineMode(){
  rai::Configuration C;
  C.addFile(rai::raiPath("../rai-robotModels/scenarios/pandaSingle.g"));

  double tau = .01;
  rai::Simulation S(C, S._physx, 2);
  Metronome tic(tau);

  //generate random waypoints
  uint T = 10;
  arr q0 = C.getJointState();
  arr q = repmat(~q0, T, 1);
  q += .5 * randn(q.d0, q.d1);
  //move command requires total time or explicit times for each control point
  double time = 10;
  S.setSplineRef(q, {time});

  for(uint t=0;t<time/tau;t++){
    tic.waitForTic();

    S.step({}, tau, S._spline);
  }

  rai::wait();
}

//===========================================================================

void testResetState(){
  rai::Configuration C;

  for(uint i=0;i<5;i++){
    rai::Frame *f = C.addFrame(STRING("block_" <<i));
    f->setShape(rai::ST_ssBox, {.2,.3,.2,.02});
    f->setColor({1,.2*i,1-.2*i});
    f->setPosition({0,0, .25*(i+1)});
    f->setMass(.1);
  }

  rai::Frame *f = C.addFrame("base");
  f->setPosition({1., 0, .5});
  f->getAts().add<bool>("multibody", true);

  f = C.addFrame("finger", "base");
  f->setShape(rai::ST_ssBox, {.3, .1, .1, .02}) .setColor({.9});
  f->setMass(.1);
  f->setJoint(rai::JT_transX);

// q0 = C.getJointState()
// X0 = C.getFrameState()

  C.view();

  double tau = .01;
  rai::Simulation S(C, S._physx, 4);
  Metronome tic(tau);
  for(uint t=0;t<2./tau;t++){
    tic.waitForTic();
    S.step({}, tau, S._none);
    C.view();
  }
}
//===========================================================================

int MAIN(int argc,char **argv){
  rai::initCmdLine(argc, argv);

  testComplexObjects();
  //testPassive("../../../../playground/24-humanoid/scene.g");
  testPassive(rai::raiPath("../rai-robotModels/scenarios/pendulum.g"));

  return 0;
}
