from robotic import BSpline
import numpy as np
import matplotlib.pyplot as plt

## B splines themselves, based on uniform knots
S = BSpline()
S.setKnots(2, np.linspace(0.,1.,10))
B = S.getBmatrix(np.linspace(0.,1.,100), False, False)

plt.plot(B)
plt.show()

## setting random control points, and fine-evaluating the resulting spline
X = np.random.randn(10,1)
S.setCtrlPoints(X)
Teval = np.linspace(-.1,1.1,100)
x = S.eval(Teval)
v = S.eval(Teval, 1)
a = S.eval(Teval, 2)

T_X = np.linspace(0.,1.,10)
plt.plot(Teval, np.hstack((x,v/10.,a/100.)))
plt.plot(T_X, X, 'p')
plt.show()

## computing optimal control points that fix the random data X from above
B = S.getBmatrix(T_X, False, True) #start not constrained to zero vel, end is
Z = np.linalg.pinv(B) @ X
S.setCtrlPoints(Z, False, True)
Y = S.eval(Teval)
plt.plot(Teval, Y)
plt.plot(T_X, X, 'p')
plt.show()
