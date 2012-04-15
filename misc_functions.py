#from potential import *

###############################################################################
"""
List of various functions needed when converting a (polynomial) potential
function given in string format to a form suitable to CUDA (i.e. C).

"""
###############################################################################

def replace_all(text, dic):
    "Use the replacement rules from dictionary dic to text string"
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def rep(f,n):
    """Write a (string) function 'f**n' into form 'f*f*...*f'
    i.e. write the multiplication open for CUDA."""
    tmp = f
    for i in xrange(n-1):
        tmp = f + '*' + tmp
    return tmp

def format_to_cuda(V,var_list,C_list,D_list,n):
    """Format a polynomial function V into a suitable form for CUDA.
    n = the degree of V."""
    tmp = V
    for f in var_list:
        for i in reversed(xrange(1,n+1)):
            tmp = tmp.replace(f+'**'+str(float(i)),rep(f,i))
            tmp = tmp.replace(f+'**'+str(i),rep(f,i))
    for C in C_list:
        for i in reversed(xrange(1,n+1)):
            tmp = tmp.replace(C+'**'+str(float(i)),rep(f,i))
            tmp = tmp.replace(C+'**'+str(i),rep(f,i))
    for D in D_list:
        for i in reversed(xrange(1,n+1)):
            tmp = tmp.replace(D+'**'+str(float(i)),rep(f,i))
            tmp = tmp.replace(D+'**'+str(i),rep(f,i))
    return tmp

def new_poly(V_s, fields, n_coeffs):
    """Make a new polynomial function that has the same powers as V_s function
       but with coefficients C1, C2..."""
    
    from sympy import Poly
    from sympy import diff, Symbol, var, simplify, sympify, S
    from sympy.core.sympify import SympifyError
    
    P = Poly(V_s,*fields)
    d = P.as_dict()
    e = {}
    
    for key in d.iterkeys():
        #print d[key], str(d[key])
        for i in xrange(1, n_coeffs+1):
            if 'C' + str(i) in str(d[key]):
                e[key] = sympify('C'+str(i))

    P2 = Poly(e,*fields)
    return str(P2.as_basic())


def V_calc(V_string, n, field_list, field_i, power_list,
           C_list, D_list, kernel_type, deriv_n=0,
           multiplier=1.0):
    """Apply the necessary calculations (i.e. derivation and/or multiply
       with a constant) to the potential function V_string:"""

    from sympy import diff, Symbol, var, simplify, sympify, S, collect
    from sympy.core.sympify import SympifyError

    n_coeffs = len(C_list)
    n2_coeffs = len(D_list)

    try:
        V = sympify(V_string)

        m = sympify(multiplier)

    except SympifyError:
        print "Could not parse expression."

    tmp = simplify(((m*diff(V, field_list[field_i-1], deriv_n)).expand()))

    tmp = collect(tmp,field_list)

    """Replace integer coefficients with float coefficients by
       first separating the different terms of the function:"""
    terms = tmp.as_coeff_factors()[1]

    """The nominator and denominator of the SymPy function have to be
       evaluated separately. Otherwise Sympy will use conjugate functions that
       will not function in CUDA code:"""
    new_terms = sympify('0')
    for term in terms:
        nom, denom = term.as_numer_denom()
        if denom == 1:
            new_terms += nom.evalf()
        else:
            new_terms += nom.evalf()/denom.evalf()

    tmp = (format_to_cuda(str(new_terms),power_list,C_list,D_list,n)
           .replace('(1 ','(1.0 '))

    if kernel_type=='H3':
        const_name='f_coeff'
        i0 = 4
        const2_name='d_coeff'
        i1 = 0
    elif kernel_type=='rp':
        const_name='g_coeff'
        i0 = 3
        const2_name='d_coeff'
        i1 = 0
    elif kernel_type=='pd':
        const_name='p_coeff'
        i0 = 2
        const2_name='d_coeff'
        i1 = 0

    "Replace C_is and D_is with the appropiate strings:"
    r_cuda_V = {}
    for i in xrange(n_coeffs):
        r_cuda_V.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
    for i in xrange(n2_coeffs):
        r_cuda_V.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})

    r_cuda_dV = {}
    for i in xrange(n_coeffs):
        r_cuda_dV.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
    for i in xrange(n2_coeffs):
        r_cuda_dV.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})

    r_cuda_d2V = {}
    for i in xrange(n_coeffs):
        r_cuda_d2V.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
    for i in xrange(n2_coeffs):
        r_cuda_d2V.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})


    if deriv_n == 0:
        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
        res = replace_all(tmp2, r_cuda_V)
    elif deriv_n == 1:
        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
        res = replace_all(tmp2, r_cuda_dV)
    elif deriv_n == 2:
        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
        res = replace_all(tmp2, r_cuda_d2V)
        
    return res

#V_string, n, field_list, field_i, n_coeffs, coeffs,
#               deriv_n=0, multiplier=1.0

def V_calc_lin(V_string, n, field_list, field_i, power_list,
               C_list, D_list, C_vals, D_vals,
               deriv_n=0, multiplier=1.0):
    
    """Apply the necessary calculations (i.e. derivation and/or multiply with a constant)
    to the potential function V_string:"""

    from sympy import diff, Symbol, var, simplify, sympify, S
    from sympy.core.sympify import SympifyError

    n_coeffs = len(C_list)
    n2_coeffs = len(D_list)

    try:
        V = sympify(V_string)

        m = sympify(multiplier)

    except SympifyError:
        print "Could not parse expression."

    #tmp = ((m*diff(V, field_list[field_i-1], deriv_n)).expand()).evalf()
    tmp = (m*diff(V, field_list[field_i-1], deriv_n)).expand()

    "Replace C_i's and D_i's with the appropiate numerical values:"
    r_V_back = {}
    for i in xrange(n_coeffs):
        r_V_back.update({'C'+str(i+1):C_vals[i]})
    for i in xrange(n2_coeffs):
        r_V_back.update({'D'+str(i+1):D_vals[i]})

    r_dV_back = {}
    for i in xrange(n_coeffs):
        r_dV_back.update({'C'+str(i+1):C_vals[i]})
    for i in xrange(n2_coeffs):
        r_dV_back.update({'D'+str(i+1):D_vals[i]})

    r_d2V_back = {}
    for i in xrange(n_coeffs):
        r_d2V_back.update({'C'+str(i+1):C_vals[i]})
    for i in xrange(n2_coeffs):
        r_d2V_back.update({'D'+str(i+1):D_vals[i]})

    f_repl = {}
    for i in xrange(len(field_list)):
        f_repl.update({'f'+str(i+1):'f0'+str(i+1)+'[0]'})

    if deriv_n == 0:
        #tmp = (tmp.subs(r_V_back)).evalf()
        tmp = tmp.subs(r_V_back)
        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
        res = replace_all(tmp2,f_repl)
    elif deriv_n == 1:
        #tmp = (tmp.subs(r_dV_back)).evalf()
        tmp = tmp.subs(r_dV_back)
        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
        res = replace_all(tmp2,f_repl)
    elif deriv_n == 2:
        #tmp = (tmp.subs(r_dV_back)).evalf()
        tmp = tmp.subs(r_dV_back)
        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
        res = replace_all(tmp2,f_repl)

    #print 'res', res
    #print 'deriv_n', deriv_n, 'res', res

    return res


def dV_coeffs(lat, V_s, field_list, C_list, deriv_n):
    """Read the different numerical coefficients
       from dV/df or d^2V/df^2 terms."""
    from sympy import diff, Symbol, var, simplify, sympify, S
    from sympy.core.sympify import SympifyError
    import numpy as np

    repl = {}
    for field in field_list:
        repl[field] = '1'

    dF = [diff(sympify(V_s),f, deriv_n) for f in field_list]

    "Make substitutions f_{i} = 1:"
    dF1 = [df.subs(repl) for df in dF]

    """Calculate the coefficients of C_{i} terms by differentiating
       with respect to them:"""
    dV_mult = []
    for Cf in dF1:
        tmp = []
        for Ci in C_list:
            tmp.append(diff(Cf,Ci))
        dV_mult.append(np.array(tmp, dtype=lat.prec_real))

    return dV_mult

###########################################################################################

def rho_init(V, fields0, pis0):
    "Calculate the energy density of the homogeneous field values:"

    from sympy import diff, Symbol, var, simplify, sympify, S, evalf
    from sympy.core.sympify import SympifyError

    C_coeff = V.C_coeff
    D_coeff = V.D_coeff
    V_string = V.V

    try:
        V = sympify(V_string)

    except SympifyError:
        print "Could not parse expression."

    rep_list = {}
    for i in xrange(len(C_coeff)):
        rep_list.update({'C'+str(i+1):C_coeff[i]})
    for i in xrange(len(D_coeff)):
        rep_list.update({'D'+str(i+1):D_coeff[i]})

    for i in xrange(len(fields0)):
        rep_list.update({'f'+str(i+1):fields0[i]})

    rho0 = 0.
    
    for i in xrange(len(fields0)):
        rho0 += 0.5*pis0[i]**2.0

    "Initial value of the potential function:"
    V0 = V.subs(rep_list)

    #print 'V0', float(V0)

    rho0 += V0

    return float(rho0)

def mass_eff(V, field_list, fields0, H0, deSitter=False):
    "Calculate the initial effective masses of the fields:"

    from sympy import diff, Symbol, var, simplify, sympify, S, evalf
    from sympy.core.sympify import SympifyError

    C_coeff = V.C_coeff
    D_coeff = V.D_coeff
    V_string = V.V

    try:
        d2V = [diff(sympify(V_string),f,2) for f in field_list]

    except SympifyError:
        print "Could not parse expression."

    "Replacement list f(t)->f(t0), C_i->C_coeff[i], D_i->D_coeff[i]:"
    rep_list = {}
    for i in xrange(len(C_coeff)):
        rep_list.update({'C'+str(i+1):C_coeff[i]})
    for i in xrange(len(D_coeff)):
        rep_list.update({'D'+str(i+1):D_coeff[i]})

    for i in xrange(len(fields0)):
        rep_list.update({'f'+str(i+1):fields0[i]})

    "If a deSitter space include also a''/a term:"
    if deSitter:
        C = 1
    else:
        C = 0
    
    "Initial value of the potential function:"
    m2eff = [(float(x.subs(rep_list)) - C*9./4.*H0**2.0 ) for x in d2V]

    for mass in m2eff:
        if mass <0:
            import sys
            print 'Mass squared negative!'
            sys.exit()


    return m2eff


def V_func(lat, V):
    "Calculate V:"

    field_list = lat.field_list

    from sympy import diff, Symbol, var, simplify, sympify, S, evalf
    from sympy.utilities import lambdify
    from sympy.core.sympify import SympifyError

    C_coeff = V.C_coeff
    D_coeff = V.D_coeff
    V_string = V.V

    try:
        V = sympify(V_string)

    except SympifyError:
        print "Could not parse expression."

    "Replacement list C_i->C_coeff[i], D_i->D_coeff[i]:"
    rep_list = {}
    for i in xrange(len(C_coeff)):
        rep_list.update({'C'+str(i+1):C_coeff[i]})
    for i in xrange(len(D_coeff)):
        rep_list.update({'D'+str(i+1):D_coeff[i]})
    
    V_func = lambdify(field_list,V.subs(rep_list))

    return V_func

def dV_func(lat, V, field_var):
    "Calculate dV/df:"

    field_list = lat.field_list

    from sympy import diff, Symbol, var, simplify, sympify, S, evalf
    from sympy.utilities import lambdify
    from sympy.core.sympify import SympifyError

    C_coeff = V.C_coeff
    D_coeff = V.D_coeff
    V_string = V.V

    try:
        dV = diff(sympify(V_string),field_var)

    except SympifyError:
        print "Could not parse expression."

    "Replacement list C_i->C_coeff[i], D_i->D_coeff[i]:"
    rep_list = {}
    for i in xrange(len(C_coeff)):
        rep_list.update({'C'+str(i+1):C_coeff[i]})
    for i in xrange(len(D_coeff)):
        rep_list.update({'D'+str(i+1):D_coeff[i]})

    dV_func = lambdify(field_list,dV.subs(rep_list))

    return dV_func

def d2V_func(lat, V, field_var):
    "Calculate the masses of the fields in terms of the field values:"

    field_list = lat.field_list

    from sympy import diff, Symbol, var, simplify, sympify, S, evalf
    from sympy.utilities import lambdify
    from sympy.core.sympify import SympifyError

    C_coeff = V.C_coeff
    D_coeff = V.D_coeff
    V_string = V.V

    try:
        d2V = diff(sympify(V_string),field_var,2)

    except SympifyError:
        print "Could not parse expression."

    "Replacement list C_i->C_coeff[i], D_i->D_coeff[i]:"
    rep_list = {}
    for i in xrange(len(C_coeff)):
        rep_list.update({'C'+str(i+1):C_coeff[i]})
    for i in xrange(len(D_coeff)):
        rep_list.update({'D'+str(i+1):D_coeff[i]})
    
    d2V_func = lambdify(field_list,d2V.subs(rep_list))

    return d2V_func

###############################################################################
# Misc functions used when writing data to file
###############################################################################

def sort_func(string):
    "Sort file names with this function:"
    start, i_value, end = string.split('.')

    return int(i_value)

def make_dir(model, lat, V, sim, path = None):
    "Create a new folder for the simulation data:"
    import os
    import datetime

    if path == None:
        path = os.getcwd()

    repl = {':':'_','.':'_','-':'_','T':'_T_'}

    time_now = datetime.datetime.now().isoformat()
    time_form = replace_all(time_now, repl)
    
    time_text = time_now.replace('T',' ')

    if model.superfolderQ:
        data_path = path + '/data/' + model.superfolder + '/' + time_form
    else:
        data_path = path + '/data/' + time_form

    os.makedirs(data_path)

    f = open(data_path + '/info.txt','w')
    f.write(create_info_file(model, lat, V, sim, time_text))
    f.close()

    return data_path

def make_subdir(path, method= None, sim_number=None):
    "Create a new subfolder for the simulation data:"
    import os

    if method == 'homog':
        data_path = path + '/homog/'
        os.makedirs(data_path)
    else:
        data_path = path + '/sim_' + str(sim_number) + '/'
        os.makedirs(data_path)

    return data_path


def create_info_file(model, lat, V, sim, time):
    "Create an info file of the simulation:"

    fields0 = [field.f0_list[0] for field in sim.fields]
    pis0 = [field.pi0_list[0] for field in sim.fields]

    text = 'PyCOOL Simulation info file\n\n'

    text += 'Simulation start time: ' + time + '\n\n'

    text += 'Simulation model: ' + V.model_name + '\n\n'

    text += 'mass m: ' + str(lat.m) + '\n'

    text += 'Reduced Planck mass: ' + str(lat.mpl) + '\n\n'

    text += 'Simulation initial time: ' + str(model.t_in*lat.m) + '/m \n'

    text += 'Simulation final time: ' + str(model.t_fin*lat.m) + '/m \n'

    text += 'Simulation initial scale parameter: ' + str(model.a_in) + '\n'

    text += ('Simulation initial radiation density: ' +
             str(sim.rho_r0/lat.m**2) + '*m**2 \n')

    text += ('Simulation initial matter density: ' +
             str(sim.rho_m0/lat.m**2) + '*m**2 \n\n')

    text += 'Is the spacetime deSitter? : ' + str(sim.deSitter) + '\n\n'

    text += 'Is debugging mode on? : ' + str(lat.test) + '\n'

    text += 'Discretization method : ' + str(lat.discQ) + '\n'

    text += 'Are gravitational waves solved? : ' + str(lat.gws) + '\n\n'

    text += 'Lattice size (n): ' + str(lat.n) + '\n'

    text += 'Lattice side length: ' + str(lat.m*lat.L) + '/m \n'

    text += 'Lattice spacing dx: ' + str(lat.m*lat.dx) + '/m \n'

    text += 'Time step d\eta: ' + str(lat.m*lat.dtau) + '/m \n\n'

    text += ('Initial field values: ' +
             ', '.join([str(x) for x in fields0])
             + '\n')

    text += ('Initial field derivative values: ' +
             ', '.join([str(x) for x in pis0])
             + '\n\n')

    if V.v_l==None:
        V_term = 'None'
    else:
        V_term = ', '.join(V.v_l)

    if V.v_int==None:
        V_int_term = 'None'
    else:
        V_int_term = ', '.join(V.v_int)

    text += ('Potential functions of the fields: ' + V_term
             + '\n')
    text += ('Interaction terms: ' + V_int_term
             + '\n')

    text += ('Numerical coefficients C_i: ' +
             ', '.join([str(x) for x in V.C_coeff])
             + '\n')
    
    text += ('Numerical coefficients D_i: ' +
             ', '.join([str(x) for x in V.D_coeff])
             + '\n\n')

    return text

def sim_time(time_sim, per_stp, steps, data_path):
    "Write simulation time into info.txt file:"
    import time, datetime

    time_str = str(datetime.timedelta(seconds=time_sim))
    per_stp_str = str(per_stp)

    sim_time = ('\nSimulation finished. Time used: ' + time_str + ' Per step: ' +
       per_stp_str + '\n')

    steps = ('Number of steps taken: ' + str(steps) + '\n')

    print sim_time

    f = open(data_path + '/info.txt','a')
    f.write(sim_time)
    f.write(steps)
    f.close()

def data_folders(path=None):
    "Give a list of possible data folders:"
    import os

    if path == None:
        path = os.getcwd() + '/data'

    folders_l = []
    for dirname, dirnames, filenames in os.walk(path):
        tmp = []
        for subdirname in dirnames:
            #folders_l.append(os.path.join(dirname, subdirname))
            tmp.append(os.path.join(dirname, subdirname))
        folders_l.append(tmp)

    return folders_l

#for item in os.listdir(data_path):
#	if os.path.isdir(os.path.join(data_path, item)):
#		print item

def sub_folders(path=None):
    "Give a list of possible data folders:"
    import os

    if path == None:
        path = os.getcwd() + '/data'

    folders_l = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folders_l.append(os.path.join(dirname, subdirname))

    return folders_l

def files_in_folder(path=None, filetype='silo', sort = True):
    "Give a list of sorted filenames in a data folder:"
    import os

    if path == None:
        path = os.getcwd() + '/data'

    #files = [path + filename for filename in os.listdir(path)]
    files = []
    for filename in os.listdir(path):
        if filetype in filename:
            files.append(path + '/' + filename)

    if sort:
        files.sort(key=sort_func)

    return files

def write_csv(lat, data_path, mode = 'non-lin', source = 'silo'):
    "This reads curves from a silo file and writes the data to a csv file:"

    import os
    import pyvisfile.silo as silo
    import csv

    if source == 'silo':

        files = files_in_folder(path=data_path, filetype='silo')

        os.makedirs(data_path + '/csv')

        print 'Writing ' + str(len(files)) +  ' cvs files.'

        i = 0
        for x in files:
            f = silo.SiloFile(x, create=False, mode=silo.DB_READ)
            curves = f.get_toc().curve_names
        
            if lat.spect and mode == 'non-lin':
                k_val = f.get_curve('field1'+'_S_k').x
            if lat.gws and mode == 'non-lin':
                k_val2 = f.get_curve('gw_spectrum').x
            if mode == 'non-lin':
                t_val = f.get_curve('a').x
            elif mode == 'homog':
                t_val = f.get_curve('a_hom').x

            f_name = data_path + '/csv/' + x.split('.')[-2] + '.csv'

            csv_file = open(f_name,'w')
            writer = csv.writer(csv_file)
            if lat.gws:
                writer.writerow(['t_val','k_val','k_val2'] + curves)
            else:
                writer.writerow(['t_val','k_val'] + curves)
            writer.writerow(t_val)
            if lat.spect and mode == 'non-lin':
                writer.writerow(k_val)
            if lat.gws and mode == 'non-lin':
                writer.writerow(k_val2)
            for curve in curves:
                writer.writerow(f.get_curve(curve).y)
        
            csv_file.close()
            f.close()

            i += 1


###############################################################################
# Misc CUDA functions
###############################################################################

def show_GPU_mem():
    import pycuda.driver as cuda

    mem_free = float(cuda.mem_get_info()[0])
    mem_free_per = mem_free/float(cuda.mem_get_info()[1])
    mem_used = float(cuda.mem_get_info()[1] - cuda.mem_get_info()[0])
    mem_used_per = mem_used/float(cuda.mem_get_info()[1])
    
    print '\nGPU memory available {0} Mbytes, {1} % of total \n'.format(
    mem_free/1024**2, 100*mem_free_per)
    
    print 'GPU memory used {0} Mbytes, {1} % of total \n'.format(
    mem_used/1024**2, 100*mem_used_per)

