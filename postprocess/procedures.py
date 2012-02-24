from jinja2 import Template

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import codecs
import numpy as np

def kernel_pd_gpu_code(lat, V, field_i, write_code=False):
    """
    Read kernel template from file and calculate"""

    kernel_name = 'kernel_pd_' + 'field' + str(field_i)

    print 'Compiling kernel: ' + kernel_name

    f = codecs.open('cuda_templates/pd_kernel.cu','r',encoding='utf-8')
    evo_code = f.read()
    f.close()

    tpl = Template(evo_code)

    fields = lat.fields

    "Write the lists of other fields:"
    other_fields = (list(set(range(1,fields+1))-set([field_i])))

    i = field_i - 1

    if field_i == 1:
        eq_sign = '='
    else:
        eq_sign = '+='

    V_term = ''
    V_i_term = ''

    if V.V_pd_i[i] != None:
        V_term += V.V_pd_i[i]
        V_i_term += V.V_pd_i[i]
        if field_i == fields:
            V_term += '+(' + V.V_pd_int + ')'
    else:
        V_i_term += '0.0'
        if field_i == fields:
            V_term += V.V_pd_int
        else:
            V_term += '0.0'

    f_code = tpl.render(kernel_name_c=kernel_name,
                        type_name_c = lat.prec_string,
                        p_coeff_l_c = V.p_coeff_l,
                        d_coeff_l_c = V.d_coeff_l,
                        field_i_c = field_i,
                        fields_c = fields,
                        block_x_c = lat.block_x2,
                        block_y_c = lat.block_y2,
                        grid_x_c = lat.grid_x,
                        grid_y_c = lat.grid_y,
                        stride_c = lat.stride,
                        DIM_X_c = lat.dimx,
                        DIM_Y_c = lat.dimy,
                        DIM_Z_c = lat.dimz,
                        other_i_c = other_fields,
                        eq_sign_c = eq_sign,
                        V_c = V_term,
                        dV2_c = V.d2V_pd[i])

    if write_code==True :
        g = codecs.open('output_kernels/debug_' + kernel_name + '.cu','w+',
                        encoding='utf-8')
        g.write(f_code)
        g.close()

    return SourceModule(f_code.encode( "utf-8" ),
                        options=['-maxrregcount='+lat.reglimit])


class Pd_Kernel:
    def __init__(self, lat, V, field_i, write_code=False):
        self.mod = kernel_pd_gpu_code(lat, V, field_i, write_code)
        self.calc = self.mod.get_function('kernel_pd_' + 'field' +
                                          str(field_i))
        self.cc_add = self.mod.get_global("c_coeff")
        self.dc_add = self.mod.get_global("d_coeff")
        self.pc_add = self.mod.get_global("p_coeff")

    def update_c(self,x):
        cuda.memcpy_htod(self.cc_add[0],x)
    def read_c(self,x):
        cuda.memcpy_dtoh(x,self.cc_add[0])

    def update_d(self,x):
        cuda.memcpy_htod(self.dc_add[0],x)
    def read_d(self,x):
        cuda.memcpy_dtoh(x,self.dc_add[0])

    def update_p(self,x):
        cuda.memcpy_htod(self.pc_add[0],x)
    def read_p(self,x):
        cuda.memcpy_dtoh(x,self.pc_add[0])



class Postprocess:
    def __init__(self, lat, V, write_code=True):

        print 'Compiling necessary post-processing kernels:'

        self.pd_kernels = [Pd_Kernel(lat, V, i, write_code)
                           for i in xrange(1,lat.fields+1)]

        "Load the discretization coefficients to constant memory:"
        for kernel in self.pd_kernels:
            kernel.update_c(lat.cc)

        for kernel in self.pd_kernels:
            kernel.update_d(V.D_coeffs_np)
            

        print "-" * 79

    def calc_pd(self, lat, V, sim):
        """Calculate m_eff^2-a''/a in conformal time.
        """

        a = sim.a
        p = sim.p
        i0 = sim.i0

        pd_args = []
        for field in sim.fields:
            pd_args.append(field.f_gpu)
        for field in sim.fields:
            pd_args.append(field.pi_gpu)

        pd_args.append(sim.pd_gpu)

        for field in sim.fields:
            pd_args.append(field.d2V_sum_gpu)

        self.pd_params = dict(block=lat.cuda_block_2,
                              grid=lat.cuda_grid,
                              stream = sim.stream)

        pd_array = V.p_coeff(lat, sim)

        for kernel in self.pd_kernels:
            kernel.update_p(pd_array)

        for kernel in self.pd_kernels:
            kernel.calc(*pd_args, **self.pd_params)

        for field in sim.fields:
            field.w_k =  (sum(sum(field.d2V_sum_gpu.get()))/lat.VL +
                          sum(sum(sim.pd_gpu.get()))/(6.*lat.VL))

            "Note that m2_eff is actually a^2*d2V/df^2."
            if lat.m2_eff:
                m2_eff = sum(sum(field.d2V_sum_gpu.get()))/lat.VL
                field.m2_eff = m2_eff
                field.m2_eff_list.append(m2_eff)

    def calc_spectrum(self, lat, V, sim):
        """Calculate the spectrums of the fields with the methods given in
           LatticeEasy or Defrost:"""

        import postprocess.calc_spect as calc

        method = lat.spect_method

        a = sim.a
        p = sim.p
        dk_inv = 1.0/lat.dk

        """a_term multiplies Pi_k, p_term F_k and a2_term multiplies
           k**2*dk**2 + a**2*m2_eff term:"""

        "Use X_k = a^(3/2)*x_k scaling:"
        #a_term = 1./a**2.
        #a2_term = 1.#1./a**2.
        #p_term = -p/(4.*lat.VL_reduced*a)
        #coeff = a**2.*lat.dx**6./(2.*lat.L**3)

        "Coefficients used in LatticeEasy i.e. use X_k = a*x_k scaling:"
        a_term = 1./a**2.
        #a2_term = 1.#1./a**2.#
        p_term = (-p/(6.*lat.VL_reduced))/a
        coeff = a**2.*lat.dx**6./(2.*lat.L**3)

        "Call calc_pd to calculate the necessary effective masses:"
        self.calc_pd(lat, V, sim)

        for field in sim.fields:
            f = field.f_gpu.get()
            Fk = np.fft.rfftn(f).transpose()

            pi_f = field.pi_gpu.get()
            Pik = np.fft.rfftn(pi_f).transpose()

            if method == 'latticeeasy':
                "LatticeEasy spectrums:"

                calc.calc_le(Fk.real, Fk.imag, Pik.real, Pik.imag,
                         field.W, field.S, field.k2_S, field.n_k,
                         field.rho_k, field.k2_rho_k, lat.dk, field.w_k,
                         a_term, p_term, coeff)

                field.S = np.where(field.W > 0.,
                               (field.S/field.W)/lat.VL**2.,
                               field.S)

                field.k2_S = np.where(field.W > 0.,
                               (field.k2_S/field.W)/lat.VL**2.,
                               field.k2_S)

                field.n_k = np.where(field.W > 0.,
                                     field.n_k/field.W,
                                     field.n_k)

                field.rho_k = 1./a*np.where(field.W > 0.,
                                     field.rho_k/field.W,
                                     field.rho_k)

                field.k2_rho_k = 1./a*np.where(field.W > 0.,
                                       field.k2_rho_k/field.W,
                                       field.k2_rho_k)

                if lat.m2_eff:
                    """Calculate the total number density of particles and
                       the fraction of particles in the relativistic
                       regime. This was explained in section 5.4 Spectra
                       in the Latticeeasy guide:"""
                
                    rel_num = 0.
                    total_N = 0.
                    total_rho = 0.

                    "Note that only non-homogeneous modes are included:"
                    for i in xrange(1,lat.ns):
                        if field.k_vals[i] > np.sqrt(field.m2_eff):
                            rel_num += (field.n_k[i]*field.W[i])
                        total_N += (field.n_k[i]*field.W[i])
                        total_rho += (field.rho_k[i]*field.W[i])

                    field.rel_num_list.append(rel_num/total_N)

                    #n_cov = total_N*(lat.dk/(2*np.pi*sim.a))**3
                    n_cov = total_N*(lat.dk/(2*np.pi))**3

                    field.n_cov_list.append(n_cov)

                    #rho_cov = total_rho*(lat.dk/(2*np.pi*sim.a))**3
                    rho_cov = total_rho*(lat.dk/(2*np.pi))**3

                    field.rho_cov_list.append(rho_cov)


            if method == 'defrost':
                "Defrost spectrums:"

                calc.calc_df(Fk.real, Fk.imag, Pik.real, Pik.imag,
                         field.W_df, field.S, field.k2_S, field.n_k,
                         field.rho_k, field.k2_rho_k, lat.dk, field.w_k,
                         a_term, p_term, coeff)

                field.S = np.where(field.W_df > 0.,
                               (field.S/field.W_df)/lat.VL**2.,
                               field.S)

                field.k2_S = np.where(field.W_df > 0.,
                               (field.k2_S/field.W_df)/lat.VL**2.,
                               field.k2_S)

                field.n_k = np.where(field.W_df > 0.,
                                     field.n_k/field.W_df,
                                     field.n_k)

                field.rho_k = 1./a*np.where(field.W_df > 0.,
                                       field.rho_k/field.W_df,
                                       field.rho_k)

                field.k2_rho_k = 1./a*np.where(field.W_df > 0.,
                                       field.k2_rho_k/field.W_df,
                                       field.k2_rho_k)

                if lat.m2_eff:
                    """Calculate the total number density of particles and
                       the fraction of particles in the relativistic
                       regime. This was explained in section 5.4 Spectra
                       in the Latticeeasy guide:"""
                
                    rel_num = 0.
                    total_N = 0.
                    total_rho = 0.

                    "Note that only non-homogeneous modes are included:"
                    for i in xrange(1,lat.ns):
                        if field.k_vals[i] > np.sqrt(field.m2_eff):
                            rel_num += (field.n_k[i]*field.W_df[i])
                        total_N += (field.n_k[i]*field.W_df[i])
                        total_rho += (field.rho_k[i]*field.W_df[i])

                    field.rel_num_list.append(rel_num/total_N)

                    #n_cov = total_N*(lat.dk/(2*np.pi*sim.a))**3
                    n_cov = total_N*(lat.dk/(2*np.pi))**3

                    field.n_cov_list.append(n_cov)

                    #rho_cov = total_rho*(lat.dk/(2*np.pi*sim.a))**3
                    rho_cov = total_rho*(lat.dk/(2*np.pi))**3

                    field.rho_cov_list.append(rho_cov)



            if method == 'k2_eff':
                "LatticeEasy spectrums:"

                calc.calc_spect_k2_eff_le(Fk.real, Fk.imag,
                                          Pik.real, Pik.imag,
                                          field.W, field.S, field.k2_S,
                                          field.n_k, field.rho_k,
                                          field.k2_rho_k, sim.k2_field,
                                          lat.dk, dk_inv, field.w_k,
                                          a_term, p_term, coeff)

                field.S = np.where(field.W > 0.,
                               (field.S/field.W)/lat.VL**2.,
                               field.S)

                field.k2_S = np.where(field.W > 0.,
                               (field.k2_S/field.W)/lat.VL**2.,
                               field.k2_S)

                field.n_k = np.where(field.W > 0.,
                                     field.n_k/field.W,
                                     field.n_k)

                field.rho_k = 1./a*np.where(field.W > 0.,
                                       field.rho_k/field.W,
                                       field.rho_k)

                field.k2_rho_k = 1./a*np.where(field.W > 0.,
                                       field.k2_rho_k/field.W,
                                       field.k2_rho_k)

                if lat.m2_eff:
                    """Calculate the total number density of particles and
                       the fraction of particles in the relativistic
                       regime. This was explained in section 5.4 Spectra
                       in the Latticeeasy guide:"""
                
                    rel_num = 0.
                    total_N = 0.
                    total_rho = 0.

                    "Note that only non-homogeneous modes are included:"
                    for i in xrange(1,len(field.k_vals)):
                        if field.k_vals[i] > np.sqrt(field.m2_eff):
                            rel_num += (field.n_k[i]*field.W[i])
                        total_N += (field.n_k[i]*field.W[i])
                        total_rho += (field.rho_k[i]*field.W[i])

                    field.rel_num_list.append(rel_num/total_N)

                    #n_cov = total_N*(lat.dk/(2*np.pi*sim.a))**3
                    n_cov = total_N*(lat.dk/(2*np.pi))**3

                    field.n_cov_list.append(n_cov)

                    #rho_cov = total_rho*(lat.dk/(2*np.pi*sim.a))**3
                    rho_cov = total_rho*(lat.dk/(2*np.pi))**3

                    field.rho_cov_list.append(rho_cov)

    def calc_dist(self, lat, sim):
        """Calculate empirical CDF and PDF functions from energy densities.
           Note that this has not been tested thoroughly:"""

        import scikits.statsmodels.tools.tools as tools

        data = sim.rho_gpu.get().flatten()/(3.*sim.H**2.)

        ecdf = tools.ECDF(data)

        x = np.linspace(min(data), max(data),500)
        cdf = ecdf(x)

        pdf = np.diff(cdf)/(x[1]-x[0])

        sim.rho_cdf = [x, cdf]

        sim.rho_pdf = [x[0:-1], pdf]

        if lat.field_rho:
            for field in sim.fields:
                data = field.rho_gpu.get().flatten()/(3.*sim.H**2.)

                ecdf = tools.ECDF(data)

                x = np.linspace(min(data), max(data), 1000)
                cdf = ecdf(x)

                pdf = np.diff(cdf)/(x[1]-x[0])

                field.rho_cdf = [x, cdf]

                field.rho_pdf = [x[0:-1], pdf]

    def calc_stats(self, lat, sim):
        """Calculate different statistical variables:"""

        import numpy as np
        import scipy.stats as stats

        for field in sim.fields:
            data = field.f_gpu.get().flatten()
            field.mean_list.append(np.mean(data))
            field.var_list.append(np.var(data))
            field.skew_list.append(stats.skew(data))
            field.kurt_list.append(stats.kurtosis(data))


    def flush(self, lat, sim, filename):
        """Write the calculated spectrums into file:"""

        import numpy as np

        mode = sim.filetype

        if mode == 'silo':
            import pyvisfile.silo as silo

            f = silo.SiloFile(filename, create=False, mode=silo.DB_APPEND)

            "Range of momentum values:"
            if lat.k2_effQ:
                k_val = np.arange(0,sim.k_bins)*lat.dk
            else:
                k_val = np.arange(0,lat.ns)*lat.dk

            options={}

            if lat.unit == 'm':
                c = 1./lat.m
                c1 = lat.m
            else:
                c = 1.
                c1 = 1.0

            #c2 = 4*np.pi*lat.dk**(-3.)

            t_val = c1*np.asarray(sim.t_write_list,dtype=np.float64)

            if lat.dist:

                f.put_curve('rho_CDF',
                            sim.rho_cdf[0],
                            sim.rho_cdf[1])

                f.put_curve('rho_PDF',
                            sim.rho_pdf[0],
                            sim.rho_pdf[1])

            n_tot = np.zeros_like(sim.fields[0].n_cov_list)

            i = 1
            for field in sim.fields:

                f.put_curve('field'+str(i)+'_S_k',
                            c*k_val,
                            c*field.S)
                f.put_curve('field'+str(i)+'_k2_S_k',
                            c*k_val,
                            c*field.k2_S)
                f.put_curve('field'+str(i)+'_n_k',
                            c*k_val,
                            c*field.n_k,
                            optlist=options)
                f.put_curve('field'+str(i)+'_rho_k',
                            c*k_val,
                            c*field.rho_k,
                            optlist=options)
                f.put_curve('field'+str(i)+'_k2_rho_k',
                            c*k_val,
                            c*field.k2_rho_k)
                f.put_curve('field'+str(i)+'_k3_rho_k',
                            c*k_val,
                            c*k_val**3.*field.rho_k)

                if lat.m2_eff:
                    m_eff_val = np.sqrt(np.asarray(field.m2_eff_list,
                                                   dtype=np.float64))

                    "Note that m2_eff_list is actually a^2*d2V/df^2:"
                    f.put_curve('field'+str(i)+'_m_eff',
                                t_val, m_eff_val/lat.m)

                    """This is the fraction of particles in relativistic
                       regime:"""

                    n_rel_val = np.asarray(field.rel_num_list,
                                           dtype=np.float64)

                    f.put_curve('field'+str(i)+'_n_rel',
                                t_val, n_rel_val)

                    """This is the comoving number density of the field:"""

                    n_cov_val = np.asarray(field.n_cov_list,
                                           dtype=np.float64)
                    f.put_curve('field'+str(i)+'_n_cov',
                                t_val, c**3*n_cov_val)

                    n_tot += n_cov_val

                    rho_cov_val = np.asarray(field.rho_cov_list,
                                           dtype=np.float64)

                    f.put_curve('field'+str(i)+'_rho_cov',
                                t_val, c**3*rho_cov_val)


                if lat.field_rho and lat.dist:
                    f.put_curve('field'+str(i)+'_rho_CDF',
                                field.rho_cdf[0],
                                field.rho_cdf[1])

                    f.put_curve('field'+str(i)+'_rho_PDF',
                                field.rho_pdf[0],
                                field.rho_pdf[1])

                if lat.stats:
                    mean_val = np.asarray(field.mean_list,
                                          dtype=np.float64)
                    f.put_curve('field'+str(i)+'_mean',
                                t_val, mean_val)
                    
                    var_val = np.asarray(field.var_list,
                                          dtype=np.float64)
                    f.put_curve('field'+str(i)+'_var',
                                t_val, var_val)

                    skew_val = np.asarray(field.skew_list,
                                          dtype=np.float64)
                    f.put_curve('field'+str(i)+'_skew',
                                t_val, skew_val)
                    
                    kurt_val = np.asarray(field.kurt_list,
                                          dtype=np.float64)
                    f.put_curve('field'+str(i)+'_kurt',
                                t_val, kurt_val)

                i += 1

            if lat.m2_eff:
                f.put_curve('n_tot', t_val, c**3*n_tot)
                

            f.close()

    def process_fields(self, lat, V, sim, filename):

            if lat.spect:
                self.calc_spectrum(lat, V, sim)

            if lat.dist:
                self.calc_dist(lat, sim)

            if lat.stats:
                self.calc_stats(lat, sim)

            self.flush(lat, sim, filename)
        

    def calc_post(self, lat, V, sim, data_path, method = 'defrost'):
        """Calculate all the required spectra, statistiscs etc. in
           folder 'data_path'. This is used when calculating spectra
           after the simulation is complete."""

        from misc_functions import files_in_folder

        files = files_in_folder(path=data_path, filetype=sim.filetype)

        print "-" * 79

        print 'Calculating the spectrums:\n'

        k = 1
        for filename in files:
            print 'Calculating spectrum ' + str(k) + ' of ' + str(len(files))

            sim.read(lat, filename)

            if lat.spect:
                self.calc_spectrum(lat, V, sim, method)

            if lat.dist:
                self.calc_dist(lat, sim)

            if lat.stats:
                self.calc_stats(lat, sim)

            self.flush(lat, sim, filename)
            
            k += 1

    def zeta_data_from_file(self, sim, model, f0_list, data_path,
                            root = False):
        """Import zeta data from files:"
           Note when importing zeta data if root = False the program
           assumes that data_path is of the default form with the
           time signature included. Set root to True if data_path
           is the root of zeta data folders."""

        import misc_functions as mf
        import os
        import csv
        import numpy as np

        if root:
            path = data_path
        else:
            path = os.path.abspath(os.path.join(data_path, os.path.pardir))

        "List of subfolders:"
        subf = mf.sub_folders(path)

        zeta_files = []

        for subfolder in subf:
            zeta_files.append(mf.files_in_folder(subfolder,
                                                 filetype='csv',
                                                 sort=False)[0])

        sim.ln_a_list = []
        sim.r_list = []

        for zeta_file in zeta_files:
            csv_file = open(zeta_file,'r')
            names=csv_file.next()

            f0_in = float(csv_file.next())
            H_ref = float(csv_file.next())
            ln_arr = np.fromstring(csv_file.next(), dtype=np.float64, sep=',')
            r_arr = np.fromstring(csv_file.next(), dtype=np.float64, sep=',')

            sim.ln_a_list.append([f0_in,H_ref,ln_arr])
            sim.r_list.append([f0_in,H_ref,r_arr])

        "Sort lists:"
        sim.ln_a_list.sort()
        sim.r_list.sort()
   
    def calc_zeta(self, sim, model, f0_list, field_i, r_decay, data_path,
                  root=False):
        """Calculates the curvature perturbation from the simulation data
           and writes the results to file. Note that this method is based on
           article 'Non-Gaussianity from resonant curvaton decay'
           http://arxiv.org/abs/0909.4535v3

           f0_list = List of different homogeneous initial values for the fields
           field_i = Index of the field which initial values are varied.
                     Should be equal to the field_i variable used in
                     field object i.e. it starts from 1."""

        import os
        import csv
        import numpy as np

        if root:
            path = data_path
        else:
            path = os.path.abspath(os.path.join(data_path, os.path.pardir))

        i = field_i - 1

        index = np.where(abs(np.array(f0_list)-model.fields0) < 1e-15)[i]

        if len(index)==0:
            print "Index of sim.fields0 in f0_list not found."
        

        ln_a = [np.array(x[2]) for x in sim.ln_a_list]
        "Matter fraction:"
        r = [np.array(x[2]) for x in sim.r_list]

        """Average fraction of matter at reference point when
           homogeneous value is equal to the one used in model file:""" 
        r_ref_ave = np.mean(r[index])

        f0_values = [x[i] for x in sim.r_list]


        "\delta ln(a) = ln(a)|_{H_ref}-ln(a*)|_{H_ref}"
        ln_a_mean = np.mean(ln_a[index])
        dln_a = ln_a - ln_a_mean

        "\delta r = r|_{H_ref} - r*|_{H_ref}"
        r_mean = np.mean(r[index])
        dr = r-r_mean
        
        "Full zeta, i.e. zeta at the different realizations:"
        zeta_full = dln_a + 0.25*(r_decay/r_ref_ave - 1.)*dr

        sim.zeta_full = zeta_full

        "Means values:"
        sim.ln_a_mean = [np.mean(x) for x in ln_a]
        sim.dln_a_mean = [np.mean(x) for x in dln_a]
        sim.r_mean = [np.mean(x) for x in r]
        sim.dr_mean = [np.mean(x) for x in dr]
        sim.zeta_mean = [np.mean(x) for x in zeta_full]

        "Standard deviations:"
        sim.ln_a_std = [np.std(x) for x in ln_a]
        sim.dln_a_std = [np.std(x) for x in dln_a]
        sim.r_std = [np.std(x) for x in r]
        sim.dr_std = [np.std(x) for x in dr]
        sim.zeta_std =  [np.std(x) for x in zeta_full]

        "Standard errors:"
        sim.ln_a_ste = [np.std(x)/np.sqrt(len(x)) for x in ln_a]
        sim.dln_a_ste = [np.std(x)/np.sqrt(len(x)) for x in dln_a]
        sim.r_ste = [np.std(x)/np.sqrt(len(x)) for x in r]
        sim.dr_ste = [np.std(x)/np.sqrt(len(x)) for x in dr]
        sim.zeta_ste =  [np.std(x)/np.sqrt(len(x)) for x in zeta_full]


        "Numpy lists:"
        f0_val = np.asarray(f0_values, dtype=np.float64)
        zeta_mean_val = np.asarray(sim.zeta_mean, dtype=np.float64)
        zeta_std_val = np.asarray(sim.zeta_std, dtype=np.float64)
        zeta_ste_val = np.asarray(sim.zeta_ste, dtype=np.float64)

        ln_a_mean_val = np.asarray(sim.ln_a_mean, dtype=np.float64)
        ln_a_std_val = np.asarray(sim.ln_a_std, dtype=np.float64)
        ln_a_ste_val = np.asarray(sim.ln_a_ste, dtype=np.float64)

        dln_a_mean_val = np.asarray(sim.dln_a_mean, dtype=np.float64)
        dln_a_std_val = np.asarray(sim.dln_a_std, dtype=np.float64)
        dln_a_ste_val = np.asarray(sim.dln_a_ste, dtype=np.float64)

        r_mean_val = np.asarray(sim.r_mean, dtype=np.float64)
        r_std_val = np.asarray(sim.r_std, dtype=np.float64)
        r_ste_val = np.asarray(sim.r_ste, dtype=np.float64)

        dr_mean_val = np.asarray(sim.dr_mean, dtype=np.float64)
        dr_std_val = np.asarray(sim.dr_std, dtype=np.float64)
        dr_ste_val = np.asarray(sim.dr_ste, dtype=np.float64)

        "Write to file (currently only silo):"
        mode = sim.filetype

        if mode == 'silo':
            import pyvisfile.silo as silo

            filename = (path + '/zeta_results'+ 'r_dec_' + str(r_decay)
                        + '.silo')

            f = silo.SiloFile(filename, mode=silo.DB_CLOBBER)

            f.put_curve('zeta_mean',f0_val,zeta_mean_val)
            f.put_curve('zeta_std',f0_val,zeta_std_val)
            f.put_curve('zeta_ste',f0_val,zeta_ste_val)
            f.put_curve('ln_a_mean',f0_val,dln_a_mean_val)
            f.put_curve('ln_a_std',f0_val,dln_a_std_val)
            f.put_curve('ln_a_ste',f0_val,dln_a_ste_val)
            f.put_curve('dln_a_mean',f0_val,dln_a_mean_val)
            f.put_curve('dln_a_std',f0_val,dln_a_std_val)
            f.put_curve('dln_a_ste',f0_val,dln_a_ste_val)
            f.put_curve('r_mean',f0_val,r_mean_val)
            f.put_curve('r_std',f0_val,r_std_val)
            f.put_curve('r_ste',f0_val,r_ste_val)
            f.put_curve('dr_mean',f0_val,dr_mean_val)
            f.put_curve('dr_std',f0_val,dr_std_val)
            f.put_curve('dr_ste',f0_val,dr_ste_val)

            f.close()

        "Write a csv file:"
        filename = (path + '/zeta_results'+ 'r_dec_' + str(r_decay)
                    + '.csv')

        csv_file = open(filename,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['f0_values','zeta_mean','zeta_std','zeta_ste',
                         'ln_a_mean','ln_a_std','ln_a_ste',
                         'dln_a_mean','dln_a_std','dln_a_ste',
                         'r_mean','r_std','r_ste','dr_mean','dr_std','dr_ste'])

        writer.writerow(f0_val)
        writer.writerow(zeta_mean_val)
        writer.writerow(zeta_std_val)
        writer.writerow(zeta_ste_val)
        writer.writerow(ln_a_mean_val)
        writer.writerow(ln_a_std_val)
        writer.writerow(ln_a_ste_val)
        writer.writerow(dln_a_mean_val)
        writer.writerow(dln_a_std_val)
        writer.writerow(dln_a_ste_val)
        writer.writerow(r_mean_val)
        writer.writerow(r_std_val)
        writer.writerow(r_ste_val)
        writer.writerow(dr_mean_val)
        writer.writerow(dr_std_val)
        writer.writerow(dr_ste_val)
        csv_file.close()


    def tensorTT_ij(self,lat, sim, u_mat,i,j):
        """Calculate the traceless-tranverse part of tensor perturbation
           component u_{ij}. u_mat defined in tensor_TT:"""

        if i == j:
            delta = 1.0
        else:
            delta = 0.0

        res = (u_mat[i][j] + 1./2.*(sim.k_vec[i]*sim.k_vec[j] -
               delta*np.ones(lat.dims_k,dtype = lat.prec_real))*
               (u_mat[0][0]+u_mat[1][1]+u_mat[2][2]) +
               01./2.*(sim.k_vec[i]*sim.k_vec[j] +
               delta*np.ones(lat.dims_k,dtype = lat.prec_real))*
               (sim.kx**2*u_mat[0][0] + sim.ky**2*u_mat[1][1] +
                sim.kz**2*u_mat[2][2] +
                2*(sim.kx*sim.ky*u_mat[0][1] +
                   sim.kx*sim.kz*u_mat[0][2] +
                   sim.ky*sim.kz*u_mat[1][2])) -
               sim.k_vec[i]*(sim.k_vec[0]*u_mat[j][0]+
                             sim.k_vec[1]*u_mat[j][1]+
                             sim.k_vec[2]*u_mat[j][2] ) -
               sim.k_vec[j]*(sim.k_vec[0]*u_mat[i][0]+
                             sim.k_vec[1]*u_mat[i][1]+
                             sim.k_vec[2]*u_mat[i][2] ))

        return res
        

    def tensor_TT(self, lat, sim, uQ = False):
        """Calculate the traceless-tranverse part of tensor perturbations
           h_{ij} and their canonical momenta pi_{u_{ij}}.
           For large lattices this will be slowish.
           If uQ = True calculate also h_{ij} which is not
           needed in the spectra:"""

        if uQ:
            u11 = sim.u11_gpu.get()
            Uk11 = np.fft.rfftn(u11).transpose()

            u12 = sim.u12_gpu.get()
            Uk12 = np.fft.rfftn(u12).transpose()

            u13 = sim.u13_gpu.get()
            Uk13 = np.fft.rfftn(u13).transpose()

            u22 = sim.u22_gpu.get()
            Uk22 = np.fft.rfftn(u22).transpose()

            u23 = sim.u23_gpu.get()
            Uk23 = np.fft.rfftn(u23).transpose()

            u33 = sim.u33_gpu.get()
            Uk33 = np.fft.rfftn(u33).transpose()

            u_mat = [[Uk11,Uk12,Uk13],
                     [Uk12,Uk22,Uk23],
                     [Uk13,Uk23,Uk33]]

            "Note that i,j = 0,1,2:"
            sim.Uk11TT = self.tensorTT_ij(lat, sim, u_mat,0,0)
            sim.Uk12TT = self.tensorTT_ij(lat, sim, u_mat,0,1)
            sim.Uk13TT = self.tensorTT_ij(lat, sim, u_mat,0,2)

            sim.Uk22TT = self.tensorTT_ij(lat, sim, u_mat,1,1)
            sim.Uk23TT = self.tensorTT_ij(lat, sim, u_mat,1,2)
        
            sim.Uk33TT = self.tensorTT_ij(lat, sim, u_mat,2,2)

        "Canonical momenta:"
        piu11 = sim.piu11_gpu.get()
        PiUk11 = np.fft.rfftn(piu11).transpose()

        piu12 = sim.piu12_gpu.get()
        PiUk12 = np.fft.rfftn(piu12).transpose()

        piu13 = sim.piu13_gpu.get()
        PiUk13 = np.fft.rfftn(piu13).transpose()

        piu22 = sim.piu22_gpu.get()
        PiUk22 = np.fft.rfftn(piu22).transpose()

        piu23 = sim.piu23_gpu.get()
        PiUk23 = np.fft.rfftn(piu23).transpose()

        piu33 = sim.piu33_gpu.get()
        PiUk33 = np.fft.rfftn(piu33).transpose()

        piu_mat = [[PiUk11,PiUk12,PiUk13],
                   [PiUk12,PiUk22,PiUk23],
                   [PiUk13,PiUk23,PiUk33]]

        "Note that i,j = 0,1,2:"
        sim.PiUk11TT = self.tensorTT_ij(lat, sim, piu_mat,0,0)
        sim.PiUk12TT = self.tensorTT_ij(lat, sim, piu_mat,0,1)
        sim.PiUk13TT = self.tensorTT_ij(lat, sim, piu_mat,0,2)

        sim.PiUk22TT = self.tensorTT_ij(lat, sim, piu_mat,1,1)
        sim.PiUk23TT = self.tensorTT_ij(lat, sim, piu_mat,1,2)
        
        sim.PiUk33TT = self.tensorTT_ij(lat, sim, piu_mat,2,2)


    def process_tensors(self, lat, sim, filename, uQ = False):
        """
        Calculate the traceless-tranverse part of tensors and
        the spectrum of graviational waves.
        Note that in the spectrum off-diagonal elements of h_ij
        are multiplied with 2 since h_ij = h_ji and only elements
        for which i>=j are calculated.
        The results are written to datafile."""

        import postprocess.calc_gw_spect as gws

        "Calculate the traceless-tranverse part of tensors:"
        self.tensor_TT(lat, sim, uQ)

        dk_inv = 1.0/lat.dk
        pi = np.pi

        "Calculate the spectra:"
        spect_k = np.zeros_like(sim.gw_spect_k)
        sim.gw_spect_k = np.zeros_like(spect_k)
        
        gws.calc_spect_pi_h(sim.PiUk11TT.real,
                            sim.PiUk11TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)

        sim.gw_spect_k += spect_k

        gws.calc_spect_pi_h(sim.PiUk12TT.real,
                            sim.PiUk12TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)
        
        sim.gw_spect_k += 2*spect_k

        gws.calc_spect_pi_h(sim.PiUk13TT.real,
                            sim.PiUk13TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)
        
        sim.gw_spect_k += 2*spect_k

        gws.calc_spect_pi_h(sim.PiUk22TT.real,
                            sim.PiUk22TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)

        sim.gw_spect_k += spect_k

        gws.calc_spect_pi_h(sim.PiUk23TT.real,
                            sim.PiUk23TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)

        sim.gw_spect_k += 2*spect_k

        gws.calc_spect_pi_h(sim.PiUk33TT.real,
                            sim.PiUk33TT.imag,
                            sim.k_abs,
                            sim.W_gw,
                            spect_k,
                            lat.dk,
                            dk_inv)

        sim.gw_spect_k += spect_k


        """Calculate the average over the bins and multiply with
           the correct coefficient:"""
        coeff = sim.a**-6*lat.mpl**2.*lat.dx**6/(8*pi**2.*lat.L**3)

        sim.gw_spect_k = coeff*np.where(sim.W_gw > 0,
                                        sim.gw_spect_k/sim.W_gw,
                                        sim.gw_spect_k)

        k_val = np.arange(0,sim.k_bins_gw)*lat.dk
        n_val = np.arange(1,sim.k_bins_gw)

        "Energy density of gravitational waves:"
        sim.rho_gw = (sim.gw_spect_k[1:]/n_val).sum()

        sim.omega_gw_list.append(sim.rho_gw/sim.rho)

        "Write to file:"
        mode = sim.filetype

        if mode == 'silo':
            import pyvisfile.silo as silo

            f = silo.SiloFile(filename, create=False, mode=silo.DB_APPEND)

            options={}

            if lat.unit == 'm':
                c = 1./lat.m
                c1 = lat.m
            else:
                c = 1.
                c1 = 1.0

            t_val = c1*np.asarray(sim.t_write_list,dtype=np.float64)
            omega_gw_val = np.asarray(sim.omega_gw_list,dtype=np.float64)
            
            f.put_curve('gw_spectrum',c*k_val,c**3.*sim.gw_spect_k)
            f.put_curve('omega_gw', t_val, omega_gw_val)

            f.close()






