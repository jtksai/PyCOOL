import pycuda.driver as cuda
import numpy as np

import integrator.symp_integrator as si
import postprocess.procedures as pp

import solvers as solv
from lattice import *

"""
###############################################################################
# Import a scalar field model and create the objects needed for simulation
###############################################################################
"""
"Necessary constants defined in the model file:"

#from models.chaotic import *
#from models.chaotic_massless import *
#from models.curvaton import *
#from models.curvaton_si import *
from models.curvaton_single import *
#from models.oscillon import *
#from models.q_ball import *

"Create a model:"
model = Model()

"Create a lattice:"
lat = Lattice(model, lin_order = 4, scale = model.scale,
              init_m = 'defrost_cpu', precision='double')

" Create a potential function object:"
V = Potential(lat, model)

"Create simulation, evolution and postprocessing instances:"
sim = si.Simulation(model, lat, V, model.a_in, model.fields0, model.pis0,
                    steps = 10000)

evo = si.Evolution(lat, V, sim)

postp = pp.Postprocess(lat, V)

if model.zetaQ == False:
    "Create a new folder for the data:"
    data_path = make_dir(model, lat, V, sim)

"""Set the average values of the fields equal to
their homogeneous values:"""
sim.adjust_fields(lat)

"""Canonical momentum p calculated with homogeneous fields.
   Field fluctuations will lead to a small perturbation into p.
   Adjust_p compensates this."""
evo.calc_rho_pres(lat, V, sim, print_Q = False, print_w=False, flush=False)
sim.adjust_p(lat)

"GPU memory status:"
show_GPU_mem()

"Time simulation:"
start = cuda.Event()
end = cuda.Event()

#start_lin = cuda.Event()
#end_lin = cuda.Event()

"""
################
Start Simulation
################
"""

"""
####################################################################
# Homogeneous evolution
####################################################################
"""

"Solve only background evolution:"
if model.homogenQ:
    print '\nSolving homogeneous equations:\n'

    solv.run_hom(lat, V, model, start, end, order = 4)

    sim.time_sim += end.time_since(start)*1e-3
    sim.per_stp += time_sim/sol.sim.i0_hom

"""
####################################################################
# Non-homogeneous evolution
####################################################################
"""

"""Run the simulations. Multiple runs can be used for
non-Gaussianity studies:"""

if model.evoQ:
    print '\nRunning ' + str(model.sim_num) + ' simulation(s):'

    solv.run_non_linear(lat, V, sim, evo, postp, model, start, end,
                        data_path, order = 4, endQ = 'time', print_Q = True,
                        print_w = False)


if sim.i0 != 0:
    "Print simulation time info:"
    sim_time(sim.time_sim, sim.per_stp, sim.i0, data_path)


"Curvature perturbation (zeta) calculations:"

if model.zetaQ:

    #r_decay = 0.0114181
    #r_decay = 0.0550699
    r_decay = 0.0369633

    "List of different homogeneous initial values for fields:"
    f0_list = [[model.f10 + i/20.*model.delta_f10/2.] for i in xrange(-20,21)]

    for fields0 in f0_list:
        solv.reinit(lat, V, sim, evo, model, model.a_in, fields0, model.pis0)
        data_path = make_dir(model, lat, V, sim)

        solv.run_non_linear(lat, V, sim, evo, postp, model, start, end,
                        data_path, order = 4, endQ = 'H', print_Q = True,
                        print_w = False)

        sim_time(sim.time_sim, sim.per_stp, sim.i0, data_path)

    postp.calc_zeta(sim, model, f0_list, 1, r_decay, data_path)






"""
####################
Simulation finished
####################
"""
print 'Done.'
