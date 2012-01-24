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

__constant__ {{ type_name_c }} c_coeff[4];
__constant__ {{ type_name_c }} g_coeff[{{ g_coeff_l_c }}];
//{% if d_coeff_l_c > 0 %} __constant__ {{ type_name_c }} d_coeff[{{ d_coeff_l_c }}]; {% endif %}
__constant__ {{ type_name_c }} d_coeff[{{ d_coeff_l_c }}];

__device__ int period(int id, int block_id, int grid_dim, int dim)
// This function will help deal with periodic boundary conditions
// when loading points to shared memory.
// id = id of a thread, block_id = id of the block,
// grid_dim = dimension of grid in numbers of blocks,
// dim = length of grid in the direction in question in numbers
// of elements.
{
    if((id < 0)&&(block_id == 0))
    {
        id = id + dim;
    } else if((id > dim-1)&&(block_id == grid_dim -1))
    {
        id = id - dim;
    }
    return id;
}

//////////////////////////////////////////////////////////////////////
// Calculate rho and pressure
//////////////////////////////////////////////////////////////////////

__global__ void {{ kernel_name_c }}({{ type_name_c }} *rho_w, {{ type_name_c }} *pres_w, {{ type_name_c }} *rho_sum_w, {{ type_name_c }} *pres_sum_w, {{ type_name_c }} *int_term_w {% for i in range(1,fields_c+1) %}, {{ type_name_c }} *field{{i}}{% endfor %}{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *pi{{i}}_m{% endfor %} {% for i in range(1,fields_c+1) %}, {{ type_name_c }} *rho_field{{i}}_sum_w{% endfor %}{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *pres_field{{i}}_sum_w {% endfor %}{% if field_rho_c == True %}{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *field{{i}}_rho_w{% endfor %}{% endif %})

{
    // field = field which energy density in being calculated.
    // field{{i}} = the other field, i in {1,2,...}
    // pi_m = canonical momentum of field
    // rho_w = total energy density of the fields
    // pres_w = total pressure of the fields
    // int_w = energy density of interaction terms
    // rho_sum_w = sum over z direction of the total energy density of the fields
    // pres_sum_w = sum over z direction of the total pressure density of the fields
    // rho_field{{ field_i_c }}_sum_w = sum over z direction of the energy density of field{{ field_i_c }}
    // pres_field{{ field_i_c }}_sum_w = sum over z direction of the pressure density of field{{ field_i_c }}


    // Shared data used in the calculation of the Laplacian of the field f
    __shared__ {{ type_name_c }} sup_data[{{ block_y_c }}][{{ block_x_c }}];
    __shared__ {{ type_name_c }} smid_data[{{ block_y_c }}][{{ block_x_c }}];
    __shared__ {{ type_name_c }} sdwn_data[{{ block_y_c }}][{{ block_x_c }}];

    // Thread ids
    // in_idx is calculated as in_idx = iy_adjusted*{{ DIM_X_c }} + ix_adjusted
    // where iy_adjusted and ix_adjusted take into accounted the periodicity
    // of the lattice
    volatile unsigned int in_idx = period(blockIdx.y*(blockDim.y-2)+threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2)+threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }});

    volatile unsigned int i0 = in_idx;

    volatile unsigned int stride = {{ stride_c }};

    {{ type_name_c }} pi{{ field_i_c }};
    {{ type_name_c }} f{{ field_i_c }};

   // Other fields needed only for the last field
   {% if field_i_c == fields_c %}
   {% for i in other_i_c %} {{ type_name_c }} f{{i}};
   {% endfor %}{% endif %}

   {% if inter_c == True %}
   {{ type_name_c }} sum_int = 0.;
   {% endif %}

    {{ type_name_c }} G, rho, pres;
    {{ type_name_c }} sum_rho_tot = 0.;
    {{ type_name_c }} sum_pres_tot = 0.;
    {{ type_name_c }} sum_rho_f = 0.;
    {{ type_name_c }} sum_pres_f = 0.;
    {{ type_name_c }} V_i, V;

    /////////////////////////////////////////
    // load the initial data into smem
    // sdwn_data from the top of the lattice
    // due to the periodicity of the lattice

    sdwn_data[threadIdx.y][threadIdx.x] = field{{ field_i_c }}[in_idx + ({{ DIM_Z_c }} - 1)*stride];
    smid_data[threadIdx.y][threadIdx.x] = field{{ field_i_c }}[in_idx];
    sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[in_idx + stride];

    __syncthreads();

    //////////////////////////////////////////////////////////
    // Calculate values only for the inner points of the block

    if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
    {
        pi{{ field_i_c }} = pi{{ field_i_c }}_m[in_idx];		// field derivative

     // Other fields needed only for the last field
        f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
   {% if field_i_c == fields_c %}
       {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
       {% endfor %}
   {% endif %}

    //////////////////////////
    // rho and pressure


    // Discretized Gradient squared operator used in rho and pressure

        G = 0.5*(c_coeff[1]*((sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[2]*((sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[3]*((sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])));

     // Calculate the necessary pressure and energy density terms:

     rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G  ;
     pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G ;

     V = {{ V_c }};
     V_i = {{ Vi_c }};

     rho_w[in_idx] {{ eq_sign_c }} rho + V;
     pres_w[in_idx] {{ eq_sign_c }} pres - V;

     {% if field_rho_c == True %}
     field{{ field_i_c }}_rho_w[in_idx] = rho + V_i;
     {% endif %}

     sum_rho_tot = rho + V;
     sum_pres_tot = pres - V;
     sum_rho_f = rho + V_i;
     sum_pres_f = pres - V_i;

     {% if inter_c == True %}
     sum_int = {{ V_int_c }};
     {% endif %}

     }

    //////////////////////////////////////////
    // advance in z direction until z={{ DIM_Z_c }}-1
    // z = {{ DIM_Z_c }}-1 calculated seperately

    volatile unsigned int i;
    for(i=1; i<({{ DIM_Z_c }}-1); i++)
    {
        in_idx  += stride;

        __syncthreads();

        ////////////////////////////////////////////////////////////
        // Update the inner up, middle and down shared memory blocks
        // by copying middle->down, up->middle and new data into up

	sdwn_data[threadIdx.y][threadIdx.x] = smid_data[threadIdx.y][threadIdx.x];
	smid_data[threadIdx.y][threadIdx.x] = sup_data[threadIdx.y][threadIdx.x];
	sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[in_idx+stride];
       
	__syncthreads();

	/////////////////////////////////////////////////////////////////
	// Calculate values only for the inner points of the thread block

	if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
	{

             pi{{ field_i_c }} = pi{{ field_i_c }}_m[in_idx];		// field derivative
            // other fields
             f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
       {% if field_i_c == fields_c %}
            {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
            {% endfor %}
       {% endif %}

        //////////////////////////////////////////////////////
        // rho and pressure


        // Discretized Gradient squared operator used in rho and pressure

               G = 0.5*(c_coeff[1]*((sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[2]*((sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[3]*((sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])));


         // Calculate the necessary pressure and energy density terms:

         rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G;
         pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G;

         V = {{ V_c }};
         V_i = {{ Vi_c }};

         rho_w[in_idx] {{ eq_sign_c }} rho + V;
         pres_w[in_idx] {{ eq_sign_c }} pres  - V;

         {% if field_rho_c == True %}
         field{{ field_i_c }}_rho_w[in_idx] = rho + V_i;
         {% endif %}

         sum_rho_tot += rho + V;
         sum_pres_tot += pres - V;
         sum_rho_f += rho + V_i;
         sum_pres_f += pres - V_i;

         {% if inter_c == True %}
         sum_int += {{ V_int_c }};
         {% endif %}

         }

     }

    //////////////////////////////////////////
    // The upper most slice of the lattice

    in_idx  += stride;

    __syncthreads();

    // Load the down, middle and up data to shared memory
    // up data now from the bottom of the lattice
    sdwn_data[threadIdx.y][threadIdx.x]  = smid_data[threadIdx.y][threadIdx.x];
    smid_data[threadIdx.y][threadIdx.x]  = sup_data[threadIdx.y][threadIdx.x];
    sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[period(blockIdx.y*(blockDim.y-2) + threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2) + threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }})];

    __syncthreads();

    if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
    {
        pi{{ field_i_c }} = pi{{ field_i_c }}_m[in_idx];		// field derivative
        // other fields
        f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
   {% if field_i_c == fields_c %}
       {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
       {% endfor %}
   {% endif %}



    //////////////////////////////////////////////////////
    // rho and pressure


    // Discretized Gradient squared operator used in rho and pressure

        G = 0.5*(c_coeff[1]*((sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[2]*((sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y][threadIdx.x]))

   + c_coeff[3]*((sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sdwn_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y-1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x-1] - smid_data[threadIdx.y][threadIdx.x])
               + (sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y+1][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x])));

     // Calculate the necessary pressure and energy density terms:

     rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G;
     pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G;

     V = {{ V_c }};
     V_i = {{ Vi_c }};

     rho_w[in_idx] {{ eq_sign_c }} rho + V;
     pres_w[in_idx] {{ eq_sign_c }} pres  - V;

     {% if field_rho_c == True %}
     field{{ field_i_c }}_rho_w[in_idx] = rho + V_i;
     {% endif %}

     sum_rho_tot += rho + V;
     sum_pres_tot += pres - V;

     sum_rho_f += rho + V_i;
     sum_pres_f += pres - V_i;

     {% if field_i_c == fields_c and inter_c == True %}
     sum_int += {{ V_int_c }};
     {% endif %}

     // Write to file:

     rho_sum_w[i0]  {{ eq_sign_c }} sum_rho_tot;
     pres_sum_w[i0]  {{ eq_sign_c }} sum_pres_tot;

     rho_field{{ field_i_c }}_sum_w[i0]  = sum_rho_f;
     pres_field{{ field_i_c }}_sum_w[i0]  = sum_pres_f;

     {% if inter_c == True %}
     int_term_w[i0] = sum_int;
     {% endif %}


    }

}

