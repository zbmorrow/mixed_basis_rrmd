import numpy as np
from rrmd_math_utils import *
import os
import multiprocessing
import functools

def mass_metric(q, drdq, M):
    drdqmat = drdq(q)
    return np.matmul(drdqmat.T, np.matmul(np.diag(M),drdqmat))

def mdhdqfun_uni(q, p, dUdQ, GQ):
    p = np.reshape(p, (p.size,1))
    q = np.reshape(q, (1,q.size))
    dUdQarray = dUdQ(q)
    h = 1e-6
    KE_curr = ke_uni(q,p,GQ)
    f = lambda z : ke_uni(z,p,GQ)
    mdHdQ = [-dirder(i, f, q, KE_curr, h) - dUdQarray[i] for i in range(p.size)]
    return np.reshape(mdHdQ,(p.size,))

def dhdpfun_uni(q, p, GQ):
    q = np.reshape(q, (1,q.size))
    p = np.reshape(p, (p.size,1))
    return np.reshape(np.linalg.solve(GQ(q),p),(p.size,))

def ke_uni(q, p, GQ):
    p = np.reshape(p,(p.size,1));
    q = np.reshape(q,(1,q.size));
    return 0.5 * np.dot(p.T, np.linalg.solve(GQ(q),p)).item()

def init_velocity(M, T):
    # use kB with units [amu * angstrom^2 * fs^(-2) * K^(-1)]
    kB = 8.31446e-7
    velocity = np.random.normal(0,1,M.shape);
    scaler = np.sqrt(kB * T) / np.sqrt(M);
    return np.multiply(scaler, velocity);

def v2p(V, M, Q, drdq):
    carP = np.reshape(np.multiply(V,M), (1,V.size))
    drdqmat = drdq(Q)
    return np.reshape(np.dot(carP, drdqmat), Q.shape)

def prep_P0(Q0, inpRec, V0=None):
    Temp = inpRec['Temp']
    M = inpRec['M']
    if type(V0) == type(None):
        V0 = init_velocity(M,Temp)
    P0 = v2p(V0, M, Q0, inpRec['dXdQ'])
    P0 = np.reshape(P0, P0.size)
    return (P0,V0)

def p2v(Q, P, inpRec, dHdPfun):
    dt = 0.001
    dQdt = dHdPfun(Q,P)
    Qmdt = Q - dt*dQdt
    Qpdt = Q + dt*dQdt
    Xmdt = inpRec['XQ'](Qmdt)
    Xpdt = inpRec['XQ'](Qpdt)
    v1d = (Xpdt - Xmdt) / (2*dt)
    v1d = np.reshape(v1d, inpRec['M'].shape)
    return v1d

def p2v_thermo(Q, P, inpRec, dHdPfun):
    v_direct = p2v(Q,P, inpRec, dHdPfun)
    v1d_reassign = init_velocity(inpRec['M'], inpRec['Temp'])
    p_interest = v2p(v1d_reassign, inpRec['M'], Q, inpRec['dXdQ'])
    v1d_interest = p2v(Q, p_interest, inpRec, dHdPfun)
    return v_direct + v1d_reassign - v1d_interest

def langevin_o_step(Q, P, inpRec, dHdPfun):
    v1d = p2v_thermo(Q, P, inpRec, dHdPfun)
    cartP = np.multiply(v1d, inpRec['M'])
    # use kB with units [amu * angstrom^2 * fs^(-2) * K^(-1)]
    kB = 8.31446e-7
    cartPnew = np.exp(-inpRec['Gamma'] * inpRec['dt']) * cartP + np.sqrt(1-np.exp(-2*inpRec['Gamma']*inpRec['dt'])) \
                * np.multiply(np.sqrt(inpRec['M'] * kB * inpRec['Temp']), np.random.normal(0,1,inpRec['M'].shape))

    cartPnew = np.reshape(cartPnew, (1, cartPnew.size))
    Pnew = np.dot(cartPnew, inpRec['dXdQ'](Q))
    return np.reshape(Pnew, P.shape)

def propagator_stormer_verlet_uni(Q0, P0, inpRec, mdHdQfun, dHdPfun, scipyDefaultSolver='hybr', forceNewton=True):
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
    P_half_true = langevin_o_step(Q_half, P_half, inpRec, dHdPfun)

    # AB
    fq = lambda Q : Q_half + 0.5 * dt * dHdPfun(Q,P_half_true) - Q
    (Qt, ithist, armijofail) = nsold(Q_half, fq, tol, maxit)
    if armijofail or (len(ithist) == maxit+1 and ithist[-1] > 1e-4):
        print('Newton didn\'t converge (P_half)!')
        return (None, None)
    Pt = P_half_true + 0.5 * dt * mdHdQfun(Qt,P_half_true)

    return (Qt, Pt)


def surfMD(Q0, P0, inpRec, chkDir, chkFreq):
    GQ = lambda Q : mass_metric(Q, inpRec['dXdQ'], inpRec['M'])
    mdHdQfun_NVE = lambda Q,P : mdhdqfun_uni(Q, P, inpRec['dUdQ'], GQ)
    dHdPfun_NVE = lambda Q,P: dhdpfun_uni(Q, P, GQ)
    maxStep = inpRec['maxStep']

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

            # outBounds = np.logical_or(Qt[N_per:] < 0.0, Qt[N_per:] > 1.0).any()
            # if outBounds:
            #     print(i,'Reassigning momenta')
            #     # if the nonperiodic variables are out of bounds, reflect momentum (diagonalized basis) for all assoc variables
            #     outinds = np.argwhere(np.logical_or(Qt[N_per:] < 0, Qt[N_per:] > 1))+N_per
            #     (V,D) = np.linalg.svd(GQ(Q0), hermitian=True)[:2]
            #     dQdt = np.linalg.solve(GQ(Q0), P0)
            #     u = np.matmul(V.T, P0)
            #     Veff = np.matmul(V,np.diag(1/D))
            #     for m in outinds:
            #         # find influential eigenvectors for relevant Q_m (whose time derivative needs to switch)
            #         # traverse u and negate until we get the right sign on dq_m/dt
            #         ordered_inds_m = np.flip(np.argsort(np.abs(Veff[m,:]))).flatten()
            #         j = 0
            #         while np.sign(np.dot(Veff[m,:], u)) == np.sign(dQdt[m]):
            #             u[ordered_inds_m[j]] = -u[ordered_inds_m[j]]
            #             j += 1
            #     P0 = np.matmul(V,u)
            #     k += 1

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
