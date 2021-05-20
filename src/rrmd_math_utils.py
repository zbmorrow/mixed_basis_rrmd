import numpy as np
import scipy.optimize

def pbcfun(x, n):
    '''
    z = pbcfun(x, n)

    Apply periodicity to inputs of polynomial interpolant manually

    Inputs:
        x:      (length-d array-like) point
        n:      (int) number of periodic variables

    Outputs:
        y:      (length-d array-like) point in interval [0,1]
    '''
    z = np.copy(x)
    z = np.reshape(z, (z.size,))
    z[:n] = z[:n] - np.floor(z[:n])
    return z

def to_canonical(q, bounds):
    '''
    y = to_canonical(q, bounds)

    Convert q from physical domain to canonical interval spacing (but
    allow for periodic extensions)

    Inputs:
        q:          (length-d array-like) point in physical domain
        bounds:     (d-by-2 array-like) endpoints for physical domain
                        intervals

    Outputs:
        y:          (length-d array-like) point in extended canonical
                        domain
    '''
    q = np.asarray(q)
    q = np.reshape(q,(q.size,))
    bounds = np.asarray(bounds)
    return np.divide(q - bounds[:,0], bounds[:,1]-bounds[:,0])

def from_canonical(q, bounds):
    '''
    y = to_canonical(q, bounds)

    Convert q from canonical interval spacing to physical domain
    (allowing for periodic extensions)

    Inputs:
        q:          (length-d array-like) point in extended canonical
                        domain
        bounds:     (d-by-2 array-like) endpoints for physical domain
                        intervals

    Outputs:
        y:          (length-d array-like) point in extended physical
                        domain
    '''
    q = np.asarray(q)
    q = np.reshape(q,(q.size,))
    bounds = np.asarray(bounds)
    return np.multiply(bounds[:,1]-bounds[:,0], q) + bounds[:,0]

def eval_mixed(x, sg_trig, sg_poly, ndof=1):
    '''
    f = eval_mixed(sg_trig, sg_poly, x, ndof)

    Evaluates a mixed interpolant constructed by both trigonometric and
    polynomial basis functions.

    INPUTS:
      sg_trig, sg_poly:
                Tasmanian structures, from grid.make*** command
      x:        (nx-by-d array) evaluation points
      ndof:     (int) number of output dimensions of interpolant,
                    changes between geom vs energy surrogate models

    OUTPUTS:
      f:        (nx-by-ndof array) evaluations of interpolant
    '''

    # first, if x is 1d array, pad into 2d array
    if len(np.shape(x)) == 1:
        x = np.reshape(x,(1,-1))

    d_trig = sg_trig.getNumDimensions()
    N_poly = int(float(sg_trig.getNumOutputs()) / float(ndof))


    # evaluate trig grid (output dimensions are polynomial grid values)
    ft = sg_trig.evaluateBatch(x[:, :d_trig])

    # get polynomial interpolation weights
    w = sg_poly.getInterpolationWeightsBatch(x[:, d_trig:])

    # do the matvec for each evaluation point
    f = np.zeros((np.shape(x)[0],ndof));
    for i in range(np.shape(x)[0]):
        M = np.reshape(ft[i,:], (ndof,N_poly)).T
        f[i,:] = np.matmul(w[i,:], M)

    if np.size(f) == 1:
        f = f.item()

    return f

def eval_mixed_grad(x, sg_trig, sg_poly, ndof=1):
    '''
    f = eval_mixed_grad(x, sg_trig, sg_poly, ndof)

    Evaluates a mixed interpolant gradient constructed by both
    trigonometric and polynomial basis functions.

    INPUTS:
      sg_trig, sg_poly:
                Tasmanian structures, from grid.make*** command
      x:        (length-d arraylike) evaluation points
      ndof:     (int) number of output dimensions of interpolant,
                    changes between geom vs energy surrogate models

    OUTPUTS:
      f:        (ndof-by-d array) Jacobian of interpolant
    '''
    h = 1e-6
    d = x.size
    x = np.reshape(x,(1,d))
    grad = np.zeros((ndof,d))
    for i in range(d):
        ei = np.zeros((1,d))
        ei[0,i] = 1
        # use centered differences since a round of forward differences will be applied on top of this
        grad[:,i] = (eval_mixed(x+h*ei,sg_trig,sg_poly,ndof)-eval_mixed(x-h*ei,sg_trig,sg_poly,ndof))/(2*h)

    if ndof == 1:
        grad = np.reshape(grad,(d,))

    return grad

def eval_grad(x, sg):
    '''
    f = eval_grad(x, sg)

    Evaluates gradient of single-basis interpolant

    Inputs:
        x:      (length-d arraylike) evaluation point
        sg:     Tasmanian structure, from grid.make*** command

    Outputs:
        f:      (ndof-by-d array) Jacobian of interpolant
    '''
    h = 1e-6
    d = sg.getNumDimensions()
    I = np.eye(d)
    x = np.reshape(x,(d,))
    xp = np.matlib.repmat(x,d,1) + h*I
    xm = np.matlib.repmat(x,d,1) - h*I
    fp = sg.evaluateBatch(xp).T
    fm = sg.evaluateBatch(xm).T
    grad = (fp - fm)/(2*h)

    if sg.getNumOutputs() == 1:
        grad = np.reshape(grad, (d,))
    return grad

def nsold(x0, f, tol=[1e-5, 1e-5], maxit=5, doArmijo=True):
    '''
    (xstar, ithist, armijofail) = nsold(x0, f, tol=[1e-5, 1e-5],
                                    maxit=5, doArmijo=True)

    Direct Newton solver with Armijo line search and 3-point parabolic
    predictor. Based on Matlab codes by C. T. Kelley
    (https://ctk.math.ncsu.edu/newton/SOLVERS/nsold.m)

    Inputs:
        x0:         (length-d arraylike) initial iterate
        f:          function
        tol:        (length-2 list) absolute and relative tolerances
        maxit:      (int, default 5) maximum number of iterations
        doArmijo:   (bool, default True) whether to do Armijo reductions

    Outputs:
        xstar:      (length-d arraylike) root of f
        ithist:     (list) ithist[n] = norm(f(x_n), 2)
        armijofail: (bool) success/failure of Armijo reduction
    '''

    def parab3p(lambdac, lambdam, ff0, ffc, ffm):
        # Translated into Python, based on original Matlab code by C. T. Kelley
        # (https://ctk.math.ncsu.edu/newton/SOLVERS/nsold.m)
        sigma0 = .1
        sigma1 = .5
        c2 = lambdam*(ffc-ff0)-lambdac*(ffm-ff0)
        if c2 >= 0:
            return sigma1*lambdac

        c1 = lambdac*lambdac*(ffm-ff0)-lambdam*lambdam*(ffc-ff0)
        lambdap = -c1 * 0.5/c2
        lambdap = sigma0*lambdac if lambdap < sigma0*lambdac else sigma1*lambdac
        return lambdap

    d = x0.size
    x = np.reshape(x0,(1,d))
    itc = 0
    f0 = f(x)
    fnrm = np.linalg.norm(f0, 2)
    ithist = [fnrm]
    stop_tol = tol[0] + fnrm*tol[1]
    armijofail = False
    while fnrm > stop_tol and itc < maxit:
        jac = np.zeros((d,d))
        for i in range(d):
            jac[:,i] = dirder(i,f,x,f0)
        direction = np.reshape(-np.linalg.solve(jac,f0.T),(d,))

        lam = 1.0
        alpha = 1.0e-4
        xt = x + lam*direction
        ft = f(xt)
        ftnrm = np.linalg.norm(ft,2)

        # do Armijo step (only necessary for regions near local maxima or saddle points)
        if doArmijo:
            iarm = 0
            maxarm = 40
            sigma1 = 0.5
            lamm = 1
            lamc = lam
            ff0 = fnrm ** 2
            ffc = ftnrm ** 2
            ffm = ffc
            while ftnrm > np.sqrt(1 - alpha*lam)*fnrm and iarm < maxarm:
                lam = sigma1*lam if iarm == 0 else parab3p(lamc, lamm, ff0, ffc, ffm)
                iarm += 1
                xt = x + lam*direction
                ft = f(xt)
                ftnrm = np.linalg.norm(ft,2)
                lamm = lamc
                lamc = lam
                ffm = ffc
                ffc = ftnrm ** 2
                if iarm == maxarm:
                    armijofail = True
                    print('Too many Armijo reductions')
                    return (np.reshape(x,x0.shape), ithist, armijofail)
        x = xt
        f0 = ft
        fnrm = ftnrm
        ithist.append(fnrm)
        itc += 1

    if itc == maxit:
        print('Maximum iterations reached')

    return (np.reshape(x,x0.shape), ithist, armijofail)

def dirder(i, f, x, f0=None, h=1e-6):
    '''
    g = dirder(i, f, x, f0=None, h=1e-6)

    Find \frac{\partial f}{\partial x_i} evaluated at x

    Inputs:
        i:      (int) gradient direction
        f:      function
        x:      (length-d arraylike) evaluation point
        f0:     (length-ndof arraylike, optional) value of f(x)
        h:      (float, default 1e-6) difference increment

    Output:
        g:      (length-ndof arraylike) gradient
    '''
    ei = np.zeros((x.size,))
    ei[i] = 1
    if type(f0) == type(None):
        f0 = f(x)
    return (f(x+h*ei) - f0) / h

def scipy_root_wrapper(x0, f, method='hybr'):
    '''
    xstar = scipy_root_wrapper(x0, f, method='hybr')

    Function of convenience. Used when solving the implicit equation
    near the periodic boundary in Stormer-Verlet integration with a
    polynomial surrogate

    Inputs:
        x0:     (length-d arraylike) initial iterate
        f:      function
        method: (str, default 'hybr') SciPy solver to use

    Outputs:
        xstar:  (length-d arraylike) root
    '''
    sol = scipy.optimize.root(f, x0, tol=1e-9, method=method)
    if not sol.success and np.linalg.norm(sol.fun,2) > 5e-6:
        print(sol.message)
        print('Residual larger than 5e-6')
        return False
    return sol.x.flatten()
