import numpy as np
from rrmd_math_utils import *
import os

def mass_metric(q, drdq, M):
    '''
    G = mass_metric(q, drdq, M)

    Computes mass-metric tensor at point q
        G_{ij} = \sum_{k=1}^{3*Natom} m_k \frac{\partial X_k (q)}
                    {\partial q_i} \frac{\partial X_k (q)}{\partial q_j}

    Inputs:
        q:      (length-d arraylike) evaluation point
        drdq:   function handle that returns X'(q)
        M:      (length-3*Natom arraylike) atomic masses

    Outputs:
        G:      (d-by-d array) mass-metric tensor
    '''
    drdqmat = drdq(q)
    return np.matmul(drdqmat.T, np.matmul(np.diag(M),drdqmat))

def mdhdqfun_uni(q, p, dUdQ, GQ):
    '''
    mdHdQ = mdhdqfun_uni(q, p, dUdQ, GQ)

    Computes -\frac{\partial \mathcal{H}}{\partial \bm q}

    Inputs:
        q:      (length-d arraylike) design variable
        p:      (length-d arraylike) generalized momentum
        dUdQ:   function handle for U'(q)
        GQ:     function handle to compute mass-metric tensor G(q)

    Outputs:
        mdHdQ:  (length-d arraylike)
    '''
    p = np.reshape(p, (p.size,1))
    q = np.reshape(q, (1,q.size))
    dUdQarray = dUdQ(q)
    h = 1e-6
    KE_curr = ke_uni(q,p,GQ)
    f = lambda z : ke_uni(z,p,GQ)
    mdHdQ = [-dirder(i, f, q, KE_curr, h) - dUdQarray[i] for i in range(p.size)]
    return np.reshape(mdHdQ,(p.size,))

def dhdpfun_uni(q, p, GQ):
    '''
    dHdP = dhdpfun_uni(q, p, GQ)

    Computes \frac{\partial \mathcal{H}}{\partial \bm p}

    Inputs:
        q:      (length-d arraylike) design variable
        p:      (length-d arraylike) generalized momentum
        GQ:     function handle to compute mass-metric tensor G(q)

    Outputs:
        dHdP:   (length-d arraylike)
    '''
    q = np.reshape(q, (1,q.size))
    p = np.reshape(p, (p.size,1))
    return np.reshape(np.linalg.solve(GQ(q),p),(p.size,))

def ke_uni(q, p, GQ):
    '''
    K = ke_uni(q, p, GQ)

    Computes kinetic energy K(q,p) = 0.5 * p^T G(q)^{-1} p

    Inputs:
        q:      (length-d arraylike) design variable
        p:      (length-d arraylike) generalized momentum
        GQ:     function handle to compute mass-metric tensor G(q)

    Outputs:
        K:      (float) kinetic energy
    '''
    p = np.reshape(p,(p.size,1));
    q = np.reshape(q,(1,q.size));
    return 0.5 * np.dot(p.T, np.linalg.solve(GQ(q),p)).item()

def init_velocity(M, T):
    '''
    V = init_velocity(M, T)

    Computes initial velocity [angstrom/fs] from Boltzmann distribution
    at temp T

    Inputs:
        M:  (length-3*Natom arraylike) atomic masses
        T:  (float) temperature

    Outputs:
        V:  (length-3*Natom arraylike) velocities
    '''
    # use kB with units [amu * angstrom^2 * fs^(-2) * K^(-1)]
    kB = 8.31446e-7
    velocity = np.random.normal(0,1,M.shape);
    scaler = np.sqrt(kB * T) / np.sqrt(M);
    return np.multiply(scaler, velocity);

def v2p(V, M, Q, drdq):
    '''
    P = v2p(V, M, Q, drdq)

    Projects velocity onto momentum by P = X'(Q)V

    Inputs:
        V:      (length-3*Natom arraylike) velocities
        M:      (length-3*Natom arraylike) atomic masses
        Q:      (length-d arraylike) design variables
        drdq:   function handle for X'(q)

    Outputs:
        P:      (length-d arraylike) generalized momentum
    '''
    carP = np.reshape(np.multiply(V,M), (1,V.size))
    drdqmat = drdq(Q)
    return np.reshape(np.dot(carP, drdqmat), Q.shape)

def prep_P0(Q0, inpRec, V0=None):
    '''
    (P0, V0) = prep_P0(Q0, inpRec, V0=None)

    Generates initial momentum

    Inputs:
        Q0:     (length-d arraylike) initial design variables
        inpRec: (dict) structure containing simulation parameters
        V0:     (length-3*Natom, optional, default None) initial
                    velocity; necessary when restarting

    Outputs:
        P0:     (length-d arraylike) initial momentum
        V0:     (length-3*Natom) initial velocity; necessary when not
                    restarting so that we can checkpoint
    '''
    Temp = inpRec['Temp']
    M = inpRec['M']
    if type(V0) == type(None):
        V0 = init_velocity(M,Temp)
    P0 = v2p(V0, M, Q0, inpRec['dXdQ'])
    P0 = np.reshape(P0, P0.size)
    return (P0,V0)

def langevin_o_step(Q, P, inpRec):
    '''
    Pnew = langevin_o_step(Q, P, inpRec)

    Computes coupling of system with thermal bath (Langevin O-step)

    Inputs:
        Q:      (length-d arraylike) design variables
        P:      (length-d arraylike) momentum
        inpRec: (dict) structure containing simulation parameters

    Outputs:
        Pnew:   (length-d arraylike) momentum with thermal effects
    '''
    # use kB with units [amu * angstrom^2 * fs^(-2) * K^(-1)]
    kB = 8.31446e-7
    Pnew = np.exp(-inpRec['Gamma'] * inpRec['dt']) * P + np.sqrt(1-np.exp(-2*inpRec['Gamma']*inpRec['dt'])) \
                * np.dot(inpRec['dXdQ'](Q).T, np.multiply(np.sqrt(inpRec['M'] * kB * inpRec['Temp']), np.random.normal(0,1,inpRec['M'].shape)))

    return np.reshape(Pnew, P.shape)

def propagator_stormer_verlet_uni(Q0, P0, inpRec, mdHdQfun, dHdPfun, scipyDefaultSolver='hybr', forceNewton=True):
    '''
    (Qt, Pt, scipyDefaultSolver) = propagator_stormer_verlet_uni(Q0, P0,
            inpRec, mdHdQfun, dHdPfun, scipyDefaultSolver='hybr',
            forceNewton=True)

    Stormer-Verlet integration scheme. Uses Newton for the implicit
    step, unless forceNewton == False and a periodic design variable is
    near the periodic boundary, in which case it uses a derivative-free
    solver in SciPy.

    Inputs:
        Q0:         (length-d arraylike) design variables at t_n
        P0:         (length-d arraylike) momenta at t_n
        mdHdQfun:   function handle for
                        -\frac{\partial \mathcal{H}(q,p)}{\partial q}
        dHdPfun:    function handle for
                        \frac{\partial \mathcal{H}(q,p)}{\partial p}
        scipyDefaultSolver:
                    (str, default 'hybr') default solver for SciPy to
                    use. Only acceptable values are 'hybr' or 'df-sane'
        forceNewton:
                    (bool, default True) whether to force the program to
                    use Newton's method

    Outputs:
        Qt:         (length-d arraylike) design variables at t_{n+1}
        Pt:         (length-d arraylike) momenta at t_{n+1}
        scipyDefaultSolver:
                    (str) encodes any change in the name of successful
                    SciPy solver, for use with all-polynomial basis only
    '''
    dof = P0.size
    Q0 = np.reshape(Q0, (dof,))
    P0 = np.reshape(P0, (dof,))
    dt = inpRec['dt']
    fq = lambda Q : Q0 + 0.5 * dt * dHdPfun(Q,P0) - Q
    # if within 2 deg of periodic boundary
    if not forceNewton and (np.any(np.abs(pbcfun(Q0, inpRec['N_per'])[:3]) < 3e-3) or np.any(np.abs(pbcfun(Q0, inpRec['N_per'])[:3] - 1.0) < 3e-3)):
        Q_half = scipy_root_wrapper(Q0, fq, scipyDefaultSolver)
        if Q_half is False:
            # switch solver algorithm
            scipyDefaultSolver = 'df-sane' if scipyDefaultSolver == 'hybr' else 'hybr'
            Q_half = scipy_root_wrapper(Q0, fq, scipyDefaultSolver)
        if Q_half is False:
            print('hybr and df-sane both failed (Q_half)')
            return (None, None, None)
    else:
        tol = np.asarray([1e-9, 1e-9])
        maxit = 10
        # aim for 1e-9
        (Q_half, ithist, armijofail) = nsold(Q0, fq, tol, maxit)
        # but don't complain unless the residual is > O(1e-6)
        if armijofail or (len(ithist) == maxit+1 and ithist[-1] > 5e-6):
            print('Newton didn\'t converge (Q_half)!')
            return (None, None, None)

    mdHdQ_half = mdHdQfun(Q_half, P0)
    fp = lambda P : P0 + 0.5 * dt * (mdHdQ_half + mdHdQfun(Q_half,P)) - P
    # same deal as before
    if not forceNewton and (np.any(np.abs(pbcfun(Q0, inpRec['N_per'])[:3]) < 3e-3) or np.any(np.abs(pbcfun(Q0, inpRec['N_per'])[:3] - 1.0) < 3e-3)):
        Pt = scipy_root_wrapper(P0, fp, scipyDefaultSolver)
        if Pt is False:
            # switch solver algorithm
            scipyDefaultSolver = 'df-sane' if scipyDefaultSolver == 'hybr' else 'hybr'
            Pt = scipy_root_wrapper(P0, fp, scipyDefaultSolver)
        if Pt is False:
            print('hybr and df-sane both failed (Pt)')
            return (None, None, None)
    else:
        tol = np.asarray([1e-9, 1e-9])
        maxit = 10
        (Pt, ithist, armijofail) = nsold(P0, fp, tol, maxit)
        if armijofail or (len(ithist) == maxit+1 and ithist[-1] > 5e-6):
            print('Newton didn\'t converge (Pt)!')
            return (None, None, None)

    Qt = Q_half + 0.5 * dt * dHdPfun(Q_half,Pt)
    return (Qt, Pt, scipyDefaultSolver)

def propagator_baoab_langevin(Q0, P0, inpRec, mdHdQfun, dHdPfun):
    '''
    (Qt, Pt) = propagator_baoab_langevin(Q0, P0, inpRec, mdHdQfun,
                    dHdPfun)

    BAOAB integration method for Langevin equations

    Inputs:
        Q0:         (length-d arraylike) design variables at t_n
        P0:         (length-d arraylike) momenta at t_n
        inpRec:     (dict) structure containing simulation parameters
        mdHdQfun:   function handle for
                        -\frac{\partial \mathcal{H}(q,p)}{\partial q}
        dHdPfun:    function handle for
                        \frac{\partial \mathcal{H}(q,p)}{\partial p}

    Outputs:
        Qt:         design variables at t_{n+1}
        Pt:         momenta at t_{n+1}
    '''
    dof = P0.size
    Q0 = np.reshape(Q0, (dof,))
    P0 = np.reshape(P0, (dof,))
    dt = inpRec['dt']
    tol = np.asarray([1e-5, 1e-5])
    maxit = 40

    # BA
    fp = lambda P : P0 + 0.5 * dt * mdHdQfun(Q0,P) - P
    (P_half, ithist, armijofail) = nsold(P0, fp, tol, maxit)
    if armijofail or (len(ithist) == maxit+1 and ithist[-1] > 1e-4):
        print('Newton didn\'t converge (P_half)!')
        return (None, None)
    Q_half = Q0 + 0.5 * dt * dHdPfun(Q0,P_half)

    # do the O-step
    P_half_true = langevin_o_step(Q_half, P_half, inpRec)

    # AB
    fq = lambda Q : Q_half + 0.5 * dt * dHdPfun(Q,P_half_true) - Q
    (Qt, ithist, armijofail) = nsold(Q_half, fq, tol, maxit)
    if armijofail or (len(ithist) == maxit+1 and ithist[-1] > 1e-4):
        print('Newton didn\'t converge (P_half)!')
        return (None, None)
    Pt = P_half_true + 0.5 * dt * mdHdQfun(Qt,P_half_true)

    return (Qt, Pt)


def surfMD(Q0, P0, inpRec, chkDir, chkFreq):
    '''
    surfMD(Q0, P0, inpRec, chkDir, chkFreq):

    Wrapper for NVE and Langevin MD on reduced-dimensional surrogate PES

    Inputs:
        Q0:         (length-d arraylike) initial design variables
        P0:         (length-d arraylike) initial momenta
        inpRec:     (dict) structure containing simulation parameters
        chkDir:     (str) checkpoint directory
        chkFreq:    (int) save outputs every chkFreq steps

    Fields of inpRec:
        restart:    (bool) whether to restart from saved trajectory
        method:     (str) 'NVE' or 'Langevin'
        N_per:      (int) number of periodic design variables
        M:          (length-3*Natom arraylike) atomic masses
        Temp:       (float) initial (NVE) or target (Langevin) temp
        dt:         (float) time step
        Tfinal:     (float) integrate up to time t = Tfinal
        bounds:     (d-by-2 arraylike) bounds for physical domain
        Gamma:      friction coefficient for Langevin equations
    '''
    GQ = lambda Q : mass_metric(Q, inpRec['dXdQ'], inpRec['M'])
    mdHdQfun_NVE = lambda Q,P : mdhdqfun_uni(Q, P, inpRec['dUdQ'], GQ)
    dHdPfun_NVE = lambda Q,P: dhdpfun_uni(Q, P, GQ)
    maxStep = int(inpRec['Tfinal'] / inpRec['dt'] + 1)

    if not inpRec['restart']:
        Qlist = []
        Plist = []
        KElist = []
        UQlist = []
        tlist = []

        Qlist.append(from_canonical(Q0,inpRec['bounds'])) # write outputs in transformed domain
        Plist.append(P0)
        KElist.append(ke_uni(Q0,P0,GQ))
        UQlist.append(inpRec['UQ'](Q0))
        tlist.append(0)
        istart = 1
        Qpred = [Q0, Q0]
        Ppred = [P0, P0]
    else:
        Qlist = np.loadtxt(os.path.join(chkDir, 'Qlist.txt')).tolist()
        Plist = np.loadtxt(os.path.join(chkDir, 'Plist.txt')).tolist()
        KElist = np.loadtxt(os.path.join(chkDir, 'KElist.txt')).tolist()
        UQlist = np.loadtxt(os.path.join(chkDir, 'UQlist.txt')).tolist()
        tlist = np.loadtxt(os.path.join(chkDir, 'tlist.txt')).tolist()
        istart = len(tlist)
        Qpred = [to_canonical(Qlist[-2],inpRec['bounds']), to_canonical(Qlist[-1], inpRec['bounds'])]
        Ppred = [np.asarray(Plist[-2]), np.asarray(Plist[-1])]

    def saveResults():
        np.savetxt(os.path.join(chkDir, 'Qlist.txt'), np.asarray(Qlist), '%.15f')
        np.savetxt(os.path.join(chkDir, 'Plist.txt'), np.asarray(Plist), '%.15f')
        np.savetxt(os.path.join(chkDir, 'KElist.txt'), np.asarray(KElist), '%.15f')
        np.savetxt(os.path.join(chkDir, 'UQlist.txt'), np.asarray(UQlist), '%.15f')
        np.savetxt(os.path.join(chkDir, 'tlist.txt'), np.asarray(tlist), '%.5f')

    scipyDefaultSolver = 'hybr'
    N_per = inpRec['N_per']
    cutoff = 1/np.sqrt(Q0.size)
    for i in range(istart, maxStep):
        k = 0
        failed = True
        forceNewton = True
        while k < 3 and failed:
            if inpRec['method'].lower() == 'nve':
                (Qt, Pt, scipyDefaultSolver) = propagator_stormer_verlet_uni(Q0, P0, inpRec, mdHdQfun_NVE, dHdPfun_NVE, scipyDefaultSolver, forceNewton)
            elif inpRec['method'].lower() == 'langevin':
                (Qt, Pt) = propagator_baoab_langevin(Q0, P0, inpRec, mdHdQfun_NVE, dHdPfun_NVE)
            else:
                print('ERROR: inpRec[\'method\'] must be either NVE or Langevin')
                return

            failed = (Qt is None)
            if failed:
                forceNewton = False
                scipyDefaultSolver = 'hybr'
                k += 1
                if k == 3:
                    print(i)
                    saveResults()
                # retry with linear predictor on components near periodic boundary (~1 deg)
                nearPB = np.logical_or(np.abs(pbcfun(Q0, inpRec['N_per'])[:3]) < 3e-3, np.abs(pbcfun(Q0, inpRec['N_per'])[:3] - 1.0) < 3e-3)
                for idx in np.argwhere(nearPB):
                    Q0[idx] = 2*Qpred[1][idx] - Qpred[0][idx]
                continue

        if failed:
            print(i,': optimizer not converged')
            saveResults()
            return

        Qlist.append(from_canonical(Qt, inpRec['bounds'])) # write outputs in transformed domain
        Plist.append(Pt)
        KElist.append(ke_uni(Qt, Pt, GQ))
        UQlist.append(inpRec['UQ'](Qt))
        tlist.append(tlist[-1] + inpRec['dt'])

        if N_per == 3 and Qt[3] > 1.1:
            print('Methyl group dissociated! Saving results...')
            saveResults()
            return

        Qpred.pop(0)
        Ppred.pop(0)
        Qpred.append(Qt)
        Ppred.append(Pt)
        Q0 = Qt
        P0 = Pt
        if i % chkFreq == 0:
            saveResults()
    saveResults()
