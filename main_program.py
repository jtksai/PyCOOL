import sys, os
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import numpy.linalg as la

import integrator.symp_integrator as si
import postprocess.spectrum as sp

from lattice import *

"""
###############################################################################
# Import a scalar field model and define a lattice
###############################################################################
"""
"Necessary constants defined in the model file:"

#from models.chaotic import *
#from models.curvaton import *
from models.curvaton_si import *
#from models.oscillon import *
#from models.q_ball import *

"Create a model:"
model = Model()

"Create a lattice:"
lat = Lattice(model, order = 4, scale = False, init_m = 'defrost_cpu')

" Create a potential function object:"
V = Potential(lat, model)

rho_fields = si.rho_field(lat, V, model.a_in, model.pis0, model.fields0)

"Create simulation, evolution and spectrum instances:"
sim = si.Simulation(model, lat, V, steps = 10000)

evo = si.Evolution(lat, V, sim)
postp = sp.Postprocess(lat, V)

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


"""
################
Start Simulation
################
"""

"""
####################################################################
# Linearized and background evolution
####################################################################
"""

"Solve only background evolution:"
if model.homogenQ:
    print '\nSolving homogeneous equations:\n'

    start.record()

    data_folder = make_subdir('homog', data_path)

    while (sim.t_hom<model.t_fin_hom):
        if (sim.i0_hom%(model.flush_freq_hom)==0):
            evo.calc_rho_pres_hom(lat, V, sim, print_Q = True)
            sim.flush_hom(lat, path = data_folder)

        sim.i0_hom += 1
        evo.evo_step_bg_2(lat, V, sim, lat.dtau_hom/sim.a_hom)

    #write_csv(lat,data_path)

    "Synchronize:"
    end.record()
    end.synchronize()

    time_sim = end.time_since(start)*1e-3
    per_stp = time_sim/sim.i0_hom


"""Solve background and linearized perturbations
   (This has not been tested thoroughly!):"""
if model.lin_evo:
    print '\nLinearized simulations:\n'

    "Save initial data:"
    evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
    #sim.flush(lat, path = data_path)

    "Go to Fourier space:"
    evo.x_to_k_space(lat, sim, perturb=True)
    evo.update(lat, sim)

    while sim.a < model.a_limit:
        "Evolve Fourier coefficients:"
        evo.lin_evo_step(lat, V, sim)
        "Evolve perturbations:"
        evo.transform(lat, sim)
        evo.calc_rho_pres_back(lat, V, sim, print_Q = True)

    "Go back to position space:"
    evo.k_to_x_space(lat, sim, unperturb=True)
    sim.adjust_fields(lat)

"""
####################################################################
# Non-linear evolution
####################################################################
"""

print '\nNon-linear simulation:\n'

"Run one simulation of non-linear evolution:"
if model.evoQ:

    start.record()
    while (sim.t<model.t_fin):
        if (sim.i0%(model.flush_freq)==0):
            evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
            data_file = sim.flush(lat, path = data_path)

            "Calculate spectrums and statistics:"
            if lat.postQ:
                postp.process_fields(lat, V, sim, data_file)

        sim.i0 += 1
        "Change this to use different order integrators:"
        evo.evo_step_2(lat, V, sim, lat.dtau)

    evo.calc_rho_pres(lat, V, sim, print_Q = True)
    data_file = sim.flush(lat, path = data_path, save_evo = False)

    "Calculate spectrums and statistics:"
    if lat.postQ:
        postp.process_fields(lat, V, sim, data_file)

    "Calculate spectrums and statistics:"
    #if lat.postQ:
    #    postp.calc_post(lat, V, sim, data_path, model.spect_m)

    end.record()
    end.synchronize()

    time_sim = end.time_since(start)*1e-3
    per_stp = time_sim/sim.i0

    write_csv(lat,data_path)

"Run multiple simulations e.g. for non-Gaussianity studies:"    
if model.nonGaussianityQ:
    print "\nRunning non-Gaussianity simulations:"

    start.record()

    i0_sum = 0

    a_list = []
    H_list = []

    for i in xrange(model.sim_num):
        print '\nSimulation run: ', i

        data_folder = make_subdir('non-gauss', data_path, i)

        while (sim.t<model.t_fin):
            if (sim.i0%(model.flush_freq)==0):
                evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
                data_file = sim.flush(lat, path = data_folder)

                "Calculate spectrums and statistics:"
                if lat.postQ:
                    postp.process_fields(lat, V, sim, data_file)

            sim.i0 += 1
            evo.evo_step_2(lat, V, sim, lat.dtau)
            
        evo.calc_rho_pres(lat, V, sim, print_Q = True)
        sim.flush(lat, path = data_folder, save_evo = False)

        i0_sum += sim.i0

        H_list.append(-np.log(np.array(sim.H_list)))
        a_list.append(np.log(np.array(sim.a_list)))

        #"Calculate spectrums and statistics:"
        #if lat.postQ:
        #    postp.calc_post(lat, V, sim, data_folder, model.spect_m)

        "Re-initialize system:"
        if i < model.sim_num-1:
            sim.reinit(model, lat, model.a_in)
            "Adjust p:"
            evo.calc_rho_pres(lat, V, sim, print_Q = False, print_w=False,
                              flush=False)
            sim.adjust_p(lat)

    #zeta_list = []
    #"Use interpolation to calculate ln(a) at H = H_homogeneous values:"
    #for i in xrange(len(a_list)):
    #    zeta_list.append(np.interp(x_hom,H_list[i],a_list[i])-y_hom)


    "Synchronize:"
    end.record()
    end.synchronize()

    time_sim = end.time_since(start)*1e-3
    per_stp = time_sim/i0_sum
            

"Print simulation time info:"
sim_time(time_sim, per_stp, sim.i0, data_path)

"""
####################
Simulation finished
####################
"""
print 'Done.'
