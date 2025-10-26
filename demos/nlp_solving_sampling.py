import robotic as ry
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def eval_fs(nlp, x):
    phi, _ = nlp.evaluate(x)
    t = np.array(nlp.getFeatureTypes())
    f = phi[t==ry.OT.f]
    r = phi[t==ry.OT.sos]
    g = phi[t==ry.OT.ineq]
    h = phi[t==ry.OT.eq]
    return np.sum(f) + np.sum(r**2), np.sum(np.maximum(g, 0)) + np.sum(np.abs(h))

def evaluateOverGrid(f, B, resolution=50):
    d = B.shape[1]
    X = d * [None]
    for i in range(d):
        X[i] = np.linspace(B[0,i], B[1,i], resolution)
    X = np.meshgrid(*X, indexing='ij')
    P = np.stack(X, axis=-1) .reshape(-1, d)
    f = np.apply_along_axis(f, 1, P)
    # f = np.array([f(p) for p in P])
    f = f.reshape( d*[resolution] )
    return f, X


def plot(nlp: ry.NLP, trace_x=None):
    B = nlp.getBounds()
    if B.size==0:
        B = np.vstack((-.5*np.ones(nlp.getDimension()), .5*np.ones(nlp.getDimension())))
    S, X = evaluateOverGrid(lambda xy: eval_fs(nlp, xy)[0], B)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(X[0],X[1],S, cmap=cm.coolwarm)
    # if trace_x is not None:
        # ax1.plot(trace_x[:,0], trace_x[:,1], trace_z, 'ko-')
    fig.colorbar(surf)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f')

    ax2 = fig.add_subplot(122)
    surf2 = plt.contourf(X[0],X[1],S, cmap=cm.coolwarm)
    if trace_x is not None:
        ax2.plot(trace_x[:,0], trace_x[:,1], 'ko-')
    fig.colorbar(surf2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
               
    plt.show()

ry.rnd_seed_random()
ry.params_clear()
ry.params_add({'Rastrigin/a': 5., 'benchmark/condition': 1.})
nlp = ry.make_NLP_Problem("RastriginSOS")
print(nlp.report(10))
# plot(nlp)

sol = ry.NLP_Solver()
sol.setProblem(nlp)
sol.setSolver(ry.OptMethod.LBFGS)
sol.solve()
trace_x = sol.getTrace_x()
print(trace_x)

plot(nlp, trace_x)