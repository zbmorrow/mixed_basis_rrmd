import numpy as np
import numpy.matlib
import os, sys
sys.path.append('../src/')
import Tasmanian as tsg    # must have Tasmanian in PYTHONPATH variable
import pandas
import time
from rrmd_core import *
import scipy.optimize


if len(sys.argv) > 2:
    Nrun = float(sys.argv[1])
    Temp = float(sys.argv[2])
elif len(sys.argv) > 1:
    Nrun = float(sys.argv[1])
    Temp = 298.15
else:
    Nrun = 1
    Temp = 298.15

print('Run index:\t',Nrun)
print('Temp:\t\t',Temp,'K')

chkDir = 'rrmd_NVE_poly_trans'
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

chkdir = chkDir + '%d' % Nrun
trigDir = 'rrmd_NVE_trig_trans/%d' % Nrun 

if not os.path.exists(chkDir):
    os.mkdir(chkDir)

# read atomic masses
state_folder = 'matfiles_allpoly_iptotal_12'
M = pandas.read_csv('%s/atom.dat' % state_folder, sep='[A-z][a-z]?', header=None, engine='python').values[:,1]
M = np.array([M]).T
N_atom = len(M)
M = np.reshape(np.matlib.repmat(M,1,3),(3*N_atom,1))

ev2ieu = 9.648e-3  # convert 1 eV to internal energy units [IEU = amu * angstrom^2 fs^(-2)]

# set input record
inpRec = dict()
inpRec['restart'] = False
inpRec['method'] = 'NVE'
inpRec['N_per'] = 3
inpRec['M'] = np.reshape(M,(M.size,))   # units = amu
inpRec['Temp'] = 298.15                 # units = K
inpRec['dt'] = 0.1                      # units = fs
inpRec['Tfinal'] = 0.5               # units = fs
inpRec['maxStep'] = int(inpRec['Tfinal'] / inpRec['dt'] + 1)
# need bounds since domains must be on same scale within the MD simulator
inpRec['bounds'] = [[-180,180], [-180,180], [-180,180], [1.1, 2.5], [90, 270]]  # units = GIC
inpRec['Gamma'] = 0.001                 # units = fs^(-1)

# PES
sg_energy = tsg.SparseGrid()
sg_energy.makeGlobalGrid(5, 1, 12, 'iptotal', 'clenshaw-curtis')
sg_energy.setDomainTransform(np.asarray([[0,1],[0,1],[0,1],[0,1],[0,1]]))
N_poly = len(sg_energy.getPoints())

dataE = ev2ieu * np.loadtxt('%s/energy.dat' % state_folder)
dataE = np.reshape(dataE,(N_poly,1))
sg_energy.loadNeededPoints(dataE)

inpRec['UQ'] = lambda Q: sg_energy.evaluate(pbcfun(Q, inpRec['N_per']))
inpRec['dUdQ'] = lambda Q: eval_grad(pbcfun(Q, inpRec['N_per']), sg=sg_energy)

# Cartesian geometry
sg_geom = tsg.SparseGrid()
sg_geom.makeGlobalGrid(5, 3*N_atom, 12, 'iptotal', 'clenshaw-curtis')
sg_geom.setDomainTransform(np.asarray([[0,1],[0,1],[0,1],[0,1],[0,1]]))

dataG = np.loadtxt('%s/geomcart.dat' % state_folder, delimiter=',')
sg_geom.loadNeededPoints(dataG)

inpRec['XQ'] = lambda Q: sg_geom.evaluate(pbcfun(Q, inpRec['N_per']))
inpRec['dXdQ'] = lambda Q: eval_grad(pbcfun(Q, inpRec['N_per']), sg=sg_geom)

if not inpRec['restart']:
    # initialize Q0 near, but not at, trans minimum on singlet PES
    Q0_guess = to_canonical([180.00, 138.85, -131.10, 1.47, 113.39], inpRec['bounds'])
    if (np.any(np.abs(pbcfun(Q0_guess, inpRec['N_per'])[:3]) < 6e-3) or np.any(np.abs(pbcfun(Q0_guess, inpRec['N_per'])[:3] - 1.0) < 6e-3)):
        Q0 = scipy.optimize.root(inpRec['dUdQ'], Q0_guess, tol=1e-5, method='hybr').x
    else:
        (Q0,ithist,armfail) = nsold(Q0_guess, inpRec['dUdQ'], [1e-5, 1e-5], 20)
    try:
        V0 = np.loadtxt(os.path.join(trigDir, 'V0.txt'))
        (P0, V0) = prep_P0(Q0, inpRec, V0)
    except:
        (P0, V0) = prep_P0(Q0, inpRec)
else:
    Q0 = to_canonical(np.loadtxt(os.path.join(chkDir, 'Qlist.txt'))[-1,:], inpRec['bounds'])
    P0 = np.loadtxt(os.path.join(chkDir, 'Plist.txt'))[-1,:]
    V0 = np.loadtxt(os.path.join(chkDir, 'V0.txt'))

# do the integration and save the output
t = time.time()
np.savetxt(os.path.join(chkDir, 'V0.txt'), V0, '%.15f')
surfMD(Q0, P0, inpRec, chkDir, 50)
print('Wall time:\t %.2f' % (time.time()-t),'sec\n')
