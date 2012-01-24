/* PyCOOL v. 0.997300203937
Copyright (C) 2011/04 Jani Sainio <jani.sainio@utu.fi>
Distributed under the terms of the GNU General Public License
http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

Please cite arXiv:
if you use this code in your research.
See also http://www.physics.utu.fi/tiedostot/theory/particlecosmology/ .

Part of this code adapted from CUDAEASY
http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
(See http://arxiv.org/abs/0911.5692 for more information.),
LATTICEEASY
http://www.science.smith.edu/departments/Physics/fstaff/gfelder/latticeeasy/
and from DEFROST http://www.sfu.ca/physics/cosmology/defrost .
(See http://arxiv.org/abs/0809.4904 for more information.)
*/

__constant__ {{ type_name_c }} h_coeff[2];

////////////////////////////////////////////////////////////////////////////////
// Scalar field evolution code - H2 equations
////////////////////////////////////////////////////////////////////////////////

__global__ void kernelH2({{ type_name_c }} *sumterm_w{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *field{{i}}{% endfor %}{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *pi{{i}}_m{% endfor %})
{
    volatile unsigned int in_idx = (blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X_c }} + (blockIdx.x*blockDim.x+threadIdx.x);
    volatile unsigned int stride = {{ stride_c }};

   {% for i in range(1,fields_c+1) %}
    //{{ type_name_c }} f{{i}}; {% endfor %}
   {% for i in range(1,fields_c+1) %}
    {{ type_name_c }} pi{{i}}; {% endfor %}

    {{ type_name_c }} sum_i = 0.;

    volatile unsigned int i;
    for(i=0; i<({{ DIM_Z_c }}); i++)
    {

    /////////////////////////////////
    //  Time step
    /////////////////////////////////

    //  Different coefficients related to equations of motion
    //  and to the chaotic potential
    //  h_coeff[0] = (dt)/(a[t]^2)
    //  h_coeff[1] = 1/(a(t)^3)*(dt)
    //  Note that when calling H2-kernel dt = d\eta/2

        {% for i in range(1,fields_c+1) %}
        
        pi{{i}} = pi{{i}}_m[in_idx];
        field{{i}}[in_idx] += h_coeff[0]*pi{{i}};
        {% endfor %}

        {{sum_c}}

        in_idx  += stride;

    }

    sumterm_w[(blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X_c }} + (blockIdx.x*blockDim.x+threadIdx.x)] {{ eq_sign_c }} h_coeff[1]*sum_i;

}

