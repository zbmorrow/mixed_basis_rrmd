import numpy as np
import numpy.matlib
import os, sys
sys.path.append('../src/') 
import Tasmanian as tsg    # must have Tasmanian in PYTHONPATH variable
import time
from rrmd_core import *


if len(sys.argv) > 3:
    Nrun = int(sys.argv[1])
    Temp = float(sys.argv[2])
    Tfinal = float(sys.argv[3])
elif len(sys.argv) > 2:
    print('md_rrSurf_mixed.py [Nrun=1] [Temp=298.15] [Tfinal=2500.0]')
    Nrun = int(sys.argv[1])
    Temp = float(sys.argv[2])
    Tfinal = 2500.0
elif len(sys.argv) > 1:
    print('md_rrSurf_mixed.py [Nrun=1] [Temp=298.15] [Tfinal=2500.0]')
    Nrun = int(sys.argv[1])
    Temp = 298.15
    Tfinal = 2500.0
else:
    print('md_rrSurf_mixed.py [Nrun=1] [Temp=298.15] [Tfinal=2500.0]')
    Nrun = 1
    Temp = 298.15
    Tfinal = 2500.0

print('  Run index: ',Nrun)
print('       Temp: ',Temp,'K')
print('     Tfinal: ',Tfinal,'fs')

chkDir = 'rrmd_NVE_mixed_trans' 
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

chkDir = chkDir + '/%d' % Nrun
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

# read atomic masses
state_folder = 'matfiles_iptotal_5_iptotal_8'
M = read_atom(state_folder)
M = np.array([M]).T
N_atom = len(M)
M = np.reshape(np.matlib.repmat(M,1,3),(3*N_atom,1))

ev2ieu = 9.648e-3  # convert 1 eV to internal energy units [IEU = amu * angstrom^2 fs^(-2)]

# set input record
inpRec = dict();
inpRec['restart'] = False
inpRec['method'] = 'NVE'
inpRec['N_per'] = 3
inpRec['M'] = np.reshape(M,(M.size,))   # units = amu
inpRec['Temp'] = Temp                   # units = K
inpRec['dt'] = 0.1                      # units = fs
inpRec['Tfinal'] = Tfinal               # units = fs
# need bounds since domains must be on same scale within the MD simulator
inpRec['bounds'] = [[-180,180], [-180,180], [-180,180], [1.1, 2.5], [90, 270]]  # units = GIC
inpRec['Gamma'] = 0.001                 # units = fs^(-1)

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
    # start at trans min on S0 surface
    Q0_guess = to_canonical([-179.71, 121.91, 121.84, 1.46, 116.06],inpRec['bounds'])
    (Q0,ithist,armfail) = nsold(Q0_guess, inpRec['dUdQ'], [1e-5, 1e-5], 50, True)
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
print('  Wall time:  %.2f' % (time.time()-t),'sec\n')
