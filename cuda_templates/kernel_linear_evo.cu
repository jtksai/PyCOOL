/* PyCOOL v. 0.997300203937
Copyright (C) 2011/04 Jani Sainio <jani.sainio@utu.fi>
Distributed under the terms of the GNU General Public License
http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

Please cite arXiv:
if you use this code in your research.
See also http://www.physics.utu.fi/tiedostot/theory/particlecosmology/pycool .

Part of this code adapted from CUDAEASY
http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
(See http://arxiv.org/abs/0911.5692 for more information.),
LATTICEEASY
http://www.science.smith.edu/departments/Physics/fstaff/gfelder/latticeeasy/
and from DEFROST http://www.sfu.ca/physics/cosmology/defrost .
(See http://arxiv.org/abs/0809.4904 for more information.)
*/

__device__ void evo_step_2({{ real_name_c }} *f01{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f0{{i}}{% endfor %}, {{ real_name_c }} *pi01{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi0{{i}}{% endfor %}, {{ real_name_c }} *f_lin_01_1{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_01_{{i}}{% endfor %}, {{ real_name_c }} *pi_lin_01_1{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_01_{{i}}{% endfor %}, {{ real_name_c }} *f_lin_10_1{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_10_{{i}}{% endfor %}, {{ real_name_c }} *pi_lin_10_1{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_10_{{i}}{% endfor %}, {{ real_name_c }} k2, {{ real_name_c }} *a, {{ real_name_c }} *b, {{ real_name_c }} *p_a, {{ real_name_c }} *t, {{ real_name_c }} deta)
// Second order symplectic integrator
// Uses pointers to register memory
{

        //{{ real_name_c }} b;


        ////////////////////////////////////
        // H1
        // Update a
        // a_coeff = -1.0/(6*V_L*m_PL^2)
        a[0] += ({{ a_coeff }})*p_a[0]*0.5*deta;

        ////////////////////////////////////
        // H2
        b[0] = 1.0/a[0];
        p_a[0] += (b[0]*b[0]*b[0]*({{ VL }})*(pi01[0]*pi01[0]{% for i in range(2,fields_c+1) %} + pi0{{i}}[0]*pi0{{i}}[0]{% endfor %}) - {{ rho_m }})*0.5*deta;

        // Background field
        {% for i in range(1,fields_c+1) %}
        f0{{i}}[0] += pi0{{i}}[0]*b[0]*b[0]*0.5*deta;{% endfor %}

        ////////////////////////////////////
        // H3

        p_a[0] += -4.0*a[0]*a[0]*a[0]*({{ V_term }})*({{ VL }})*0.5*deta;

        // Background momentum field
        {% for i in range(1,fields_c+1) %}
        pi0{{i}}[0] -= a[0]*a[0]*a[0]*a[0]*({{ dV[i-1] }})*0.5*deta;{% endfor %}

        t[0] += a[0]*0.5*deta;

        ////////////////////////////////////////////////
        // Evolve perturbations

        // Perturbation field evolution
        // For f_lin_10i initially f_lin_10i = 1.0 and pi_lin_10i = 0.0
        // and vice versa for f_lin_11i and pi_lin_11i
        {% for i in range(1,fields_c+1) %}
        f_lin_01_{{i}}[0] += pi_lin_01_{{i}}[0]*b[0]*b[0]*0.5*deta;
        f_lin_10_{{i}}[0] += pi_lin_10_{{i}}[0]*b[0]*b[0]*0.5*deta;
        {% endfor %}

        // Perturbation field evolution
        // For f_lin_10i initially f_lin_10i = 1.0 and pi_lin_10i = 0.0
        // and vice versa for f_lin_11i and pi_lin_11i

        {% for i in range(1,fields_c+1) %}
        pi_lin_01_{{i}}[0] -= a[0]*a[0]*(k2 + a[0]*a[0]*({{ d2V0[i-1] }}))*f_lin_01_{{i}}[0]*deta;
        pi_lin_10_{{i}}[0] -= a[0]*a[0]*(k2 + a[0]*a[0]*({{ d2V1[i-1] }}))*f_lin_10_{{i}}[0]*deta;
        {% endfor %}

        // Perturbation field evolution
        // For f_lin_10i initially f_lin_10i = 1.0 and pi_lin_10i = 0.0
        // and vice versa for f_lin_11i and pi_lin_11i

        {% for i in range(1,fields_c+1) %}
        f_lin_01_{{i}}[0] += pi_lin_01_{{i}}[0]*b[0]*b[0]*0.5*deta;
        f_lin_10_{{i}}[0] += pi_lin_10_{{i}}[0]*b[0]*b[0]*0.5*deta;
        {% endfor %}

        ////////////////////////////////////
        // H3

        p_a[0] += -4.0*a[0]*a[0]*a[0]*({{ V_term }})*({{ VL }})*0.5*deta;

        // Background momentum field
        {% for i in range(1,fields_c+1) %}
        pi0{{i}}[0] -= a[0]*a[0]*a[0]*a[0]*({{ dV[i-1] }})*0.5*deta;{% endfor %}

        t[0] += a[0]*0.5*deta;
        ////////////////////////////////////
        // H2

        p_a[0] += (b[0]*b[0]*b[0]*({{ VL }})*(pi01[0]*pi01[0]{% for i in range(2,fields_c+1) %} + pi0{{i}}[0]*pi0{{i}}[0]{% endfor %}) - {{ rho_m }})*0.5*deta;

        // Background field
        {% for i in range(1,fields_c+1) %}
        f0{{i}}[0] += pi0{{i}}[0]*b[0]*b[0]*0.5*deta;{% endfor %}

        ////////////////////////////////////
        // H1
        // Update a
        a[0] += {{ a_coeff }}*p_a[0]*0.5*deta;

}



//////////////////////////////////////////////////////////////////////
// Linearized perturbation evolution solver
//////////////////////////////////////////////////////////////////////
// This kernel is used to evolve linearized equations with two different initial values:
// for dfield_i_10 and dpi_i_10_m initially dfield = 1 and dpi_i = 0
// and similarly for dfield_i_01 and dpi_i_01_m initially dfield = 0 and dpi_i = 1.

__global__ void linear_evo({{ real_name_c }} *f01_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f0{{i}}_m{% endfor %}, {{ real_name_c }} *pi01_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi0{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_10_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_10_{{i}}_m{% endfor %}, {{ real_name_c }} *a_val, {{ real_name_c }} *p_a_val, {{ real_name_c }} *t_val, {{ real_name_c }} deta, int steps, {{ real_name_c }} *k2_bins)

{

    volatile unsigned int in_idx = blockIdx.x*blockDim.x+threadIdx.x;

    volatile unsigned int i;
    {{ real_name_c }} a, p_a, k2, t;
    {{ real_name_c }} b;

    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} f0{{i}};{% endfor %}
    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} pi0{{i}};{% endfor %}
    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} f_lin_01_{{i}};{% endfor %}
    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} f_lin_10_{{i}};{% endfor %}
    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} pi_lin_01_{{i}};{% endfor %}
    {% for i in range(1,fields_c+1) %}
    {{ real_name_c }} pi_lin_10_{{i}};{% endfor %}

    // Include only non-homogeneous (k != 0) modes:
    k2 = k2_bins[in_idx];

    // Set initial values

    {% for i in range(1,fields_c+1) %}
    f0{{i}} = f0{{i}}_m[0];
    pi0{{i}} = pi0{{i}}_m[0];

    f_lin_01_{{i}} = 0.0;
    f_lin_10_{{i}} = 1.0;

    pi_lin_01_{{i}} = 1.0;
    pi_lin_10_{{i}} = 0.0;
    {% endfor %}

    a = a_val[0];
    p_a = p_a_val[0];
    t = t_val[0];


    __syncthreads();

    /////////////////////////////////////////
    // Calculations

    /////////////////////////////////
    //  Evolution step
    /////////////////////////////////

    for(i=0; i<(steps); i++)
    {
    {% if order == 2 %}
        evo_step_2(&f01{% for i in range(2,fields_c+1) %}, &f0{{i}}{% endfor %}, &pi01{% for i in range(2,fields_c+1) %}, &pi0{{i}}{% endfor %}, &f_lin_01_1{% for i in range(2,fields_c+1) %}, &f_lin_01_{{i}}{% endfor %}, &pi_lin_01_1{% for i in range(2,fields_c+1) %}, &pi_lin_01_{{i}}{% endfor %}, &f_lin_10_1{% for i in range(2,fields_c+1) %}, &f_lin_10_{{i}}{% endfor %}, &pi_lin_10_1{% for i in range(2,fields_c+1) %}, &pi_lin_10_{{i}}{% endfor %}, k2, &a, &b, &p_a, &t, deta);
    {% else %}
    {% for w in w_i %}
        evo_step_2(&f01{% for i in range(2,fields_c+1) %}, &f0{{i}}{% endfor %}, &pi01{% for i in range(2,fields_c+1) %}, &pi0{{i}}{% endfor %}, &f_lin_01_1{% for i in range(2,fields_c+1) %}, &f_lin_01_{{i}}{% endfor %}, &pi_lin_01_1{% for i in range(2,fields_c+1) %}, &pi_lin_01_{{i}}{% endfor %}, &f_lin_10_1{% for i in range(2,fields_c+1) %}, &f_lin_10_{{i}}{% endfor %}, &pi_lin_10_1{% for i in range(2,fields_c+1) %}, &pi_lin_10_{{i}}{% endfor %}, k2, &a, &b, &p_a, &t, {{w}}*deta);{% endfor %}
    {%endif %}

    }

    // Write final values
    // Commented writes were used for debugging
    {% for i in range(1,fields_c+1) %}

    f_lin_01_{{i}}_m[in_idx] = f_lin_01_{{i}};
    f_lin_10_{{i}}_m[in_idx] = f_lin_10_{{i}};

    pi_lin_01_{{i}}_m[in_idx] = pi_lin_01_{{i}};
    pi_lin_10_{{i}}_m[in_idx] = pi_lin_10_{{i}};
    {% endfor %}

    //k2_field[in_idx] = k2;

    if((threadIdx.x==0)&&(threadIdx.y==0)&&(blockIdx.x==0)&&(blockIdx.y==0))
    {
        a_val[0] = a;
        p_a_val[0] = p_a;
        t_val[0] = t;

        {% for i in range(1,fields_c+1) %}
        f0{{i}}_m[0] = f0{{i}};
        pi0{{i}}_m[0] = pi0{{i}};
        {% endfor %}
    }

}

