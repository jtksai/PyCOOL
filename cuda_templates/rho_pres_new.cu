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

__constant__ {{ type_name_c }} c1_coeff[2];
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
    {% set radius2 = 2*radius_c %}
    {% set radiusp1 = radius_c + 1%}{% set radiusm1 = radius_c - 1%}{% set radiusp2 = radius_c + 2%}{% set radiusm2 = radius_c - 2%}
    {% set radiusp3 = radius_c + 3%}{% set radiusm3 = radius_c - 3%}{% set radiusp4 = radius_c + 4%}{% set radiusm4 = radius_c - 4%}

    {% set blockIdx01 = DIM_X_c - radius_c %}
    {% set blockIdx02 = radius_c%}
    {% set blockIdx11 = DIM_X_c - block_x_c%}
    {% set blockIdx12 = block_x_c%}

    {% set blockIdy01 = DIM_X_c*(DIM_Y_c - radius_c) %}
    {% set blockIdy02 = DIM_X_c*radius_c %}
    {% set blockIdy11 = DIM_X_c*(DIM_Y_c - block_y_c) %}
    {% set blockIdy12 = DIM_X_c*(block_y_c) %}

    {% set gridx1 =  grid_x_c - 1 %}
    {% set gridy1 =  grid_y_c - 1 %}

    {% set blockx1 =  block_x_c + radius_c %}
    {% set blocky1 =  block_y_c + radius_c %}

    {% set down_idx =  stride_c*(DIM_Z_c - 1)%}
    // Shared data used in the calculation of the Laplacian of the field f
    __shared__ {{ type_name_c }} s_data[{{ block_y_c }} + {{ radius2 }}][{{ block_x_c }} + {{ radius2 }}];

    // Thread ids
    // in_idx is used to load data into the top of the stencil
    // out_idx is used to load data into the shared memory
    volatile unsigned int out_idx = {{ DIM_X_c }}*(blockIdx.y*blockDim.y + threadIdx.y) + blockIdx.x*blockDim.x + threadIdx.x;
    volatile unsigned int in_idx = out_idx + {{ down_idx }};
    volatile unsigned int i,j;

    {{ type_name_c }} pi{{ field_i_c }};
    {{ type_name_c }} f{{ field_i_c }};

    {{ type_name_c }} up[{{ radius_c }}];
    {{ type_name_c }} down[{{ radius_c }}];

   // Other fields needed only for the last field
   {% if field_i_c == fields_c %}
   {% for i in other_i_c %} {{ type_name_c }} f{{i}};
   {% endfor %}{% endif %}

   {% if inter_c == True %}
   {{ type_name_c }} sum_int = 0.;
   {% endif %}

    {{ type_name_c }} Dxf, Dyf, Dzf;
    {{ type_name_c }} G, rho, pres;
    {{ type_name_c }} sum_rho_tot = 0.;
    {{ type_name_c }} sum_pres_tot = 0.;
    {{ type_name_c }} sum_rho_f = 0.;
    {{ type_name_c }} sum_pres_f = 0.;
    {{ type_name_c }} V_i, V;

    /////////////////////////////////////////
    // load the initial data into shared mem
    // down data from the top of the lattice
    // due to the periodicity of the lattice

    // Down data
    // In a multi-gpu implementation these values could be loaded from a different device
//#pragma unroll {{ radiusp1 }}
    for (j = 0; j < {{ radius_c }}; j++)
    {
        down[j] = field{{ field_i_c }}[in_idx];
        in_idx -= {{ stride_c }};
    }

    //  Inner points of shared memory
    s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx];

    // West halo
    if (blockIdx.x == 0){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx + ({{ blockIdx01 }})];
        }
    }
    else {
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx - ({{ blockIdx02 }})];
        }
    }
    // East halo
    if (blockIdx.x == {{ gridx1 }}){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.x < {{ radius_c }}){
            //sumterm_w[out_idx] = out_idx -({{ blockIdx11 }});
            s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx - ({{ blockIdx11 }})];
        }
    }
    else {
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx + {{ blockIdx12 }}];
        }
    }
    // North halo
    if (blockIdx.y == 0){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy01 }}];
        }
    }
    else {
        if (threadIdx.y < {{ radius_c }}){
            //sumterm_w[out_idx] = out_idx - ({{ blockIdy02 }});
            s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy02 }})];
         }
    }

    // South halo
    if (blockIdx.y == {{ gridy1 }}){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded fjennifer lopez heightrom a different device
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy11 }})];
        }
    }
    else {
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy12 }}];
         }
    }

    // Up data
    // In a multi-gpu implementation these values could be loaded from a different device
    in_idx = out_idx + {{ stride_c }};

//#pragma unroll {{ radius_c }}
    for (j = 0 ; j < {{ radius_c }} ; j++)
    {
        up[j] = field{{ field_i_c }}[in_idx];
        in_idx += {{ stride_c }};
    }
    
    __syncthreads();

    f{{ field_i_c }} = s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}]; 	// current field value
    pi{{ field_i_c }} = pi{{ field_i_c }}_m[out_idx];		// field derivative

    // other fields needed only for the last field
   {% if field_i_c == fields_c %}
       {% for i in other_i_c %} f{{i}} = field{{i}}[out_idx];
       {% endfor %}
   {% endif %}

    //////////////////////////
    // rho and pressure


    // Discretized Gradient squared operator used in rho and pressure

        {% if method_c == "hlattice" and radius_c == 4%}
        Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]) + 
              c1_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm2 }})]);
        Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]) + 
              c1_coeff[1]*(s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm2 }})][threadIdx.x + {{ radius_c }}]);
        Dzf = c1_coeff[0]*(up[0] - down[0]) + 
              c1_coeff[1]*(up[1] - down[1]);

        {% elif method_c == "latticeeasy"%}

        Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]);
        Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]);
        Dzf = c1_coeff[0]*(up[0] - down[0]);

        {% endif %}





     G = Dxf*Dxf + Dyf*Dyf +Dzf*Dzf;

     // Calculate the necessary pressure and energy density terms:

     rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G  ;
     pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G ;

     V = {{ V_c }};
     V_i = {{ Vi_c }};

     rho_w[out_idx] {{ eq_sign_c }} rho + V;
     pres_w[out_idx] {{ eq_sign_c }} pres - V;

     {% if field_rho_c == True %}
     field{{ field_i_c }}_rho_w[out_idx] = rho + V_i;
     {% endif %}

     sum_rho_tot = rho + V;
     sum_pres_tot = pres - V;
     sum_rho_f = rho + V_i;
     sum_pres_f = pres - V_i;

     {% if inter_c == True %}
     sum_int = {{ V_int_c }};
     {% endif %}

    {% set foo = DIM_Z_c-radius_c %}
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // advance in z direction until z={{ foo }}
    // {{ foo }} <= z < {{ DIM_Z_c }} calculated seperately

//#pragma unroll {{ foo }}
    for(i=1; i<({{ foo }}); i++)
    {
        __syncthreads();

        // Advance the slice (move the thread-front)
//#pragma unroll {{ radiusm1 }}
        for (int j = {{ radiusm1 }} ; j > 0 ; j--)
            down[j] = down[j - 1];

        down[0] = s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}];
        s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}] = up[0];
        f{{ field_i_c }} = up[0]; 	// current field value

//#pragma unroll {{ radiusm1 }}
        for (int j = 0 ; j < {{ radiusm1 }}; j++)
            up[j] = up[j + 1];
        up[{{ radiusm1 }}] = field{{ field_i_c }}[in_idx];

        in_idx += {{ stride_c }};
        out_idx += {{ stride_c }};

        // West halo
        if (blockIdx.x == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx + ({{ blockIdx01 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx - ({{ blockIdx02 }})];
            }
        }
        // East halo
        if (blockIdx.x == {{ gridx1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx - ({{ blockIdx11 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx + {{ blockIdx12 }}];
            }
        }
        // North halo
        if (blockIdx.y == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy01 }}];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy02 }})];
             }
        }
        // South halo
        if (blockIdx.y == {{ gridy1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy11 }})];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy12 }}];
             }
        }
       
	__syncthreads();

        // field derivative
        pi{{ field_i_c }} = pi{{ field_i_c }}_m[out_idx];
        // other fields needed only for the last field
       {% if field_i_c == fields_c %}
           {% for i in other_i_c %} f{{i}} = field{{i}}[out_idx];
           {% endfor %}
       {% endif %}

        //////////////////////////////////////////////////////
        // rho and pressure


        // Discretized Gradient squared operator used in rho and pressure

            {% if method_c == "hlattice" and radius_c == 4%}
            Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                               s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]) + 
                  c1_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] -
                               s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm2 }})]);
            Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                               s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]) + 
                  c1_coeff[1]*(s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] -
                               s_data[threadIdx.y + ({{ radiusm2 }})][threadIdx.x + {{ radius_c }}]);
            Dzf = c1_coeff[0]*(up[0] - down[0]) + 
                  c1_coeff[1]*(up[1] - down[1]);

            {% elif method_c == "latticeeasy"%}

            Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                               s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]);
            Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                               s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]);
            Dzf = c1_coeff[0]*(up[0] - down[0]);

            {% endif %}

         G = Dxf*Dxf + Dyf*Dyf +Dzf*Dzf;

         // Calculate the necessary pressure and energy density terms:

         rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G;
         pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G;

         V = {{ V_c }};
         V_i = {{ Vi_c }};

         rho_w[out_idx] {{ eq_sign_c }} rho + V;
         pres_w[out_idx] {{ eq_sign_c }} pres  - V;

         {% if field_rho_c == True %}
         field{{ field_i_c }}_rho_w[out_idx] = rho + V_i;
         {% endif %}

         sum_rho_tot += rho + V;
         sum_pres_tot += pres - V;
         sum_rho_f += rho + V_i;
         sum_pres_f += pres - V_i;

         {% if inter_c == True %}
         sum_int += {{ V_int_c }};
         {% endif %}

    }


    {% set foo = DIM_Z_c-radius_c %}
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // up data now from the bottom of the lattice due to periodicity
    //

    in_idx = {{ DIM_X_c }}*(blockIdx.y*blockDim.y + threadIdx.y) + blockIdx.x*blockDim.x + threadIdx.x;

{% if radius_c > 1%}
#pragma unroll {{ radius_c }}
{% endif %}
    for(i={{ foo }}; i<({{ DIM_Z_c }}); i++)
    {
        __syncthreads();

        // Advance the slice (move the thread-front)
{% if radiusm1 > 0%}
#pragma unroll {{ radiusm1 }}
        for (j = {{ radiusm1 }} ; j > 0 ; j--)
            down[j] = down[j - 1];
{% endif %}

        down[0] = s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}];
        s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}] = up[0];
        f{{ field_i_c }} = up[0]; 	// current field value


{% if radius_c > 1%}
#pragma unroll {{ radius_c }}
        for (j = 0 ; j < {{ radiusm1 }}; j++)
            up[j] = up[j + 1];
{% endif %}
        up[{{ radiusm1 }}] = field{{ field_i_c }}[out_idx];

        in_idx  += {{ stride_c }};
        out_idx  += {{ stride_c }};



        // West halo
        if (blockIdx.x == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx + ({{ blockIdx01 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = field{{ field_i_c }}[out_idx - ({{ blockIdx02 }})];
            }
        }
        // East halo
        if (blockIdx.x == {{ gridx1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx - ({{ blockIdx11 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = field{{ field_i_c }}[out_idx + {{ blockIdx12 }}];
            }
        }
        // North halo
        if (blockIdx.y == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy01 }}];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy02 }})];
             }
        }
        // South halo
        if (blockIdx.y == {{ gridy1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx - ({{ blockIdy11 }})];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = field{{ field_i_c }}[out_idx + {{ blockIdy12 }}];
             }
        }
       
	__syncthreads();

        // field derivative
        pi{{ field_i_c }} = pi{{ field_i_c }}_m[out_idx];
        // other fields needed only for the last field
       {% if field_i_c == fields_c %}
           {% for i in other_i_c %} f{{i}} = field{{i}}[out_idx];
           {% endfor %}
       {% endif %}

        //////////////////////////////////////////////////////
        // rho and pressure

        // Discretized Gradient squared operator used in rho and pressure

        {% if method_c == "hlattice" and radius_c == 4%}
        Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]) + 
              c1_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm2 }})]);
        Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]) + 
              c1_coeff[1]*(s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm2 }})][threadIdx.x + {{ radius_c }}]);
        Dzf = c1_coeff[0]*(up[0] - down[0]) + 
              c1_coeff[1]*(up[1] - down[1]);

        {% elif method_c == "latticeeasy"%}

        Dxf = c1_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] -
                           s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + ({{ radiusm1 }})]);
        Dyf = c1_coeff[0]*(s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] -
                           s_data[threadIdx.y + ({{ radiusm1 }})][threadIdx.x + {{ radius_c }}]);
        Dzf = c1_coeff[0]*(up[0] - down[0]);

        {% endif %}

        G = Dxf*Dxf + Dyf*Dyf +Dzf*Dzf;

        // Calculate the necessary pressure and energy density terms:

        rho = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[1]*G;
        pres = g_coeff[0]*pi{{ field_i_c }}*pi{{ field_i_c }} + g_coeff[2]*G;

        V = {{ V_c }};
        V_i = {{ Vi_c }};

        rho_w[out_idx] {{ eq_sign_c }} rho + V;
        pres_w[out_idx] {{ eq_sign_c }} pres  - V;

        {% if field_rho_c == True %}
        field{{ field_i_c }}_rho_w[out_idx] = rho + V_i;
        {% endif %}

        sum_rho_tot += rho + V;
        sum_pres_tot += pres - V;

        sum_rho_f += rho + V_i;
        sum_pres_f += pres - V_i;

        {% if field_i_c == fields_c and inter_c == True %}
        sum_int += {{ V_int_c }};
        {% endif %}

    }

     // Write to file:

     out_idx = {{ DIM_X_c }}*(blockIdx.y*blockDim.y + threadIdx.y) + blockIdx.x*blockDim.x + threadIdx.x;

     rho_sum_w[out_idx] {{ eq_sign_c }} sum_rho_tot;
     pres_sum_w[out_idx] {{ eq_sign_c }} sum_pres_tot;

     rho_field{{ field_i_c }}_sum_w[out_idx]  = sum_rho_f;
     pres_field{{ field_i_c }}_sum_w[out_idx]  = sum_pres_f;

     {% if inter_c == True %}
     int_term_w[out_idx] = sum_int;
     {% endif %}

}

