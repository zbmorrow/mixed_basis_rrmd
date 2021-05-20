import numpy as np
import numpy.matlib
import os, sys
sys.path.append('../src/')
import Tasmanian as tsg    # must have Tasmanian in PYTHONPATH variable
import pandas
import time
from rrmd_core import *

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

chkDir = 'rrmd_langevin_trig_triplet_torsion'
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

chkDir = chkDir + '/%d' % Nrun
if not os.path.exists(chkDir):
    os.mkdir(chkDir)


# read atomic masses
state_folder = 'matfiles_iptotal_5_iptotal_8_triplet'
M = pandas.read_csv('%s/atom.dat' % state_folder, sep='[A-z][a-z]?', header=None, engine='python').values[:,1]
M = np.array([M]).T
N_atom = len(M)
M = np.reshape(np.matlib.repmat(M,1,3),(3*N_atom,1))

ev2ieu = 9.648e-3  # convert 1 eV to internal energy units [IEU = amu * angstrom^2 fs^(-2)]

# set input record
inpRec = dict();
inpRec['restart'] = False
inpRec['method'] = 'langevin'
inpRec['N_per'] = 3
inpRec['M'] = np.reshape(M,(M.size,))   # units = amu
inpRec['Temp'] = Temp                   # units = K
inpRec['dt'] = 0.05                     # units = fs
inpRec['Tfinal'] = 40000.0              # units = fs
# need bounds since domains must be on same scale within the MD simulator
inpRec['bounds'] = [[-180,180], [-180,180], [-180,180], [1.1, 2.5], [90, 270]]  # units = GIC
inpRec['Gamma'] = 0.01                  # units = fs^(-1)

# make sparse grids for PES
sg_poly_energy = tsg.SparseGrid()
sg_poly_energy.makeGlobalGrid(2,1,8,'iptotal','clenshaw-curtis')
sg_poly_energy.setDomainTransform(np.asarray([[0, 1], [0,1]]))
N_poly = len(sg_poly_energy.getPoints())

sg_trig_energy = tsg.SparseGrid()
sg_trig_energy.makeFourierGrid(3,N_poly,5,'iptotal',[1, 2, 2])
sg_trig_energy.setDomainTransform(np.asarray([[0, 1], [0,1], [0, 1]]))
N_trig = len(sg_trig_energy.getPoints())

dataE = ev2ieu * np.loadtxt('%s/energy.dat' % state_folder)
dataE = np.reshape(dataE,(N_trig,N_poly))
sg_trig_energy.loadNeededPoints(dataE)

inpRec['UQ'] = lambda Q : eval_mixed(Q, sg_trig=sg_trig_energy, sg_poly=sg_poly_energy, ndof=1)
inpRec['dUdQ'] = lambda Q : eval_mixed_grad(Q, sg_trig=sg_trig_energy, sg_poly=sg_poly_energy, ndof=1)


# make sparse grids for Cartesian geometry
sg_poly_geom = tsg.SparseGrid()
sg_poly_geom.makeGlobalGrid(2,1,8,'iptotal','clenshaw-curtis')
sg_poly_geom.setDomainTransform(np.asarray([[0, 1], [0,1]]))

sg_trig_geom = tsg.SparseGrid()
sg_trig_geom.makeFourierGrid(3,N_poly*N_atom*3,5,'iptotal',[1, 2, 2])
sg_trig_geom.setDomainTransform(np.asarray([[0, 1], [0,1], [0, 1]]))

# each row of geomcart.dat is formatted like
#
#    X_1^{poly node 1}, ..., X_{30}^{poly node 1}, ..., X_1^{poly node N_poly}, ..., X_{30}^{poly node N_poly}
#
# where the row index corresponds to the trig node index
dataG = np.loadtxt('%s/geomcart.dat' % state_folder,delimiter=',')
sg_trig_geom.loadNeededPoints(dataG)

inpRec['XQ'] = lambda Q : eval_mixed(Q, sg_trig=sg_trig_geom, sg_poly=sg_poly_geom, ndof=3*N_atom)
inpRec['dXdQ'] = lambda Q : eval_mixed_grad(Q, sg_trig=sg_trig_geom, sg_poly=sg_poly_geom, ndof=3*N_atom)

# get initial position, velocity, and momentum
if not inpRec['restart']:
    # initialize Q0 at trans min on singlet PES
    Q0_guess = to_canonical([-96.26, 127.57, 68.63, 1.48, 128.24],inpRec['bounds'])
    # # initialize to torsion minimum on triplet PES
    # Q0_guess = to_canonical([-96.26, 127.57, 68.63, 1.48, 128.24], inpRec['bounds'])
    (Q0,ithist,armfail) = nsold(Q0_guess, inpRec['dUdQ'], [1e-3, 1e-3], 50, True)
    try:
        V0 = np.loadtxt(os.path.join(chkDir, 'V0.txt'))
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
surfMD(Q0, P0, inpRec, chkDir, 500)
print('Wall time:\t %.2f' % (time.time()-t),'sec\n')
