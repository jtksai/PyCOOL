/* PyCOOL v. 0.997300203937
Copyright (C) 2011/04 Jani Sainio <jani.sainio@utu.fi>
Distributed under the terms of the GNU General Public License
http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

Please cite arXiv:
if you use this code in your research.
See also http://www.physics.utu.fi/tiedostot/theory/particlecosmology/pycool/ .

Part of this code adapted from CUDAEASY
http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
(See http://arxiv.org/abs/0911.5692 for more information.),
LATTICEEASY
http://www.science.smith.edu/departments/Physics/fstaff/gfelder/latticeeasy/
and from DEFROST http://www.sfu.ca/physics/cosmology/defrost .
(See http://arxiv.org/abs/0809.4904 for more information.)
*/


__constant__ {{ type_name_c }} c1_coeff[2];
__constant__ {{ type_name_c }} c_coeff[4];
__constant__ {{ type_name_c }} f_coeff[{{ f_coeff_l_c }}];
//{% if d_coeff_l_c > 0 %} __constant__ {{ type_name_c }} d_coeff[{{ d_coeff_l_c }}]; {% endif %}
__constant__ {{ type_name_c }} d_coeff[{{ d_coeff_l_c }}];

////////////////////////////////////////////////////////////////////////////////
// Scalar field evolution
////////////////////////////////////////////////////////////////////////////////

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

__device__ double atomicAdd(double* address, double val)
// Double precision atomic add function
{
    double old = *address, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(
                     atomicCAS((unsigned long long int*)address,
                                __double_as_longlong(assumed),
                                __double_as_longlong(val + assumed)));
        } while (assumed != old);
     return old;
}

//////////////////////////////////////////////////////////////////////
// Scalar field evolution code - H3 equations
//////////////////////////////////////////////////////////////////////

__global__ void {{ kernel_name_c }}({{ type_name_c }} *sumterm_w{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *field{{i}}{% endfor %}{% for i in range(1,fields_c+1) %}, {{ type_name_c }} *pi{{i}}_m{% endfor %} {% if gw_c %}, {{ type_name_c }} *piu11_m, {{ type_name_c }} *piu12_m, {{ type_name_c }} *piu22_m, {{ type_name_c }} *piu13_m, {{ type_name_c }} *piu23_m, {{ type_name_c }} *piu33_m{% endif %})

{

    // Shared data used in the calculation of the Laplacian of the field f
    __shared__ {{ type_name_c }} sup_data[{{ block_y_c }}][{{ block_x_c }}];
    __shared__ {{ type_name_c }} smid_data[{{ block_y_c }}][{{ block_x_c }}];
    __shared__ {{ type_name_c }} sdwn_data[{{ block_y_c }}][{{ block_x_c }}];

    // Thread ids
    // in_idx is calculated as in_idx = iy_adjusted*{{ DIM_X_c }} + ix_adjusted
    // where iy_adjusted and ix_adjusted take into accounted the periodicity
    // of the lattice
    volatile unsigned int in_idx = period(blockIdx.y*(blockDim.y-2)+threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2)+threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }});
    //volatile unsigned int stride = {{ stride_c }};

    {{ type_name_c }} f{{ field_i_c }};
    //{{ type_name_c }} pi{{ field_i_c }};
   {% for i in other_i_c %} {{ type_name_c }} f{{i}};
   {% endfor %}
    {{ type_name_c }} D2f;
    {{ type_name_c }} sumi=0.;

    //{% if gw_c %}
    //{{ type_name_c }} Dxf, Dyf, Dzf;
    //{% endif %}

    {% if tmp_c %}
   {% for i in range(1,trms_c+1) %} {{ type_name_c }} tmp{{i}};{% endfor %}
    {% endif %}

    /////////////////////////////////////////
    // load the initial data into smem
    // sdwn_data from the top of the lattice
    // due to the periodicity of the lattice

    sdwn_data[threadIdx.y][threadIdx.x] = field{{ field_i_c }}[in_idx + ({{ DIM_Z_c }} - 1)*{{ stride_c }}];
    smid_data[threadIdx.y][threadIdx.x] = field{{ field_i_c }}[in_idx];
    sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[in_idx + {{ stride_c }}];

    __syncthreads();

    //////////////////////////////////////////////////////////
    // Calculate values only for the inner points of the block
    {% set blockx =  block_x_c -1 %}
    {% set blocky =  block_y_c -1 %}

    if(((threadIdx.x>0)&&(threadIdx.x<({{ blockx }})))&&((threadIdx.y>0)&&(threadIdx.y<({{ blocky }}))))
    {
        f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
        //pi{{ field_i_c }} = pi{{ field_i_c }}_m[in_idx];		// field derivative
        // other fields
       {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
       {% endfor %}

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator
    // f_coeff[1] = dt*a(t)/(dx^2)
    // c_coeff's = laplacian discretization coefficients

        D2f = f_coeff[1]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
              + c_coeff[1]*( sdwn_data[threadIdx.y][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x+1]
              + smid_data[threadIdx.y-1][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x] + sup_data[threadIdx.y][threadIdx.x])
              + c_coeff[2]*(sdwn_data[threadIdx.y][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x] 
              + sdwn_data[threadIdx.y][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x]
              + smid_data[threadIdx.y-1][threadIdx.x+1] + smid_data[threadIdx.y-1][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x-1] + smid_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x]
              + sup_data[threadIdx.y][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x] )
              + c_coeff[3]*( sdwn_data[threadIdx.y-1][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x-1]
              + sdwn_data[threadIdx.y+1][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y-1][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x-1]
              + sup_data[threadIdx.y+1][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x+1]));


    /////////////////////////////////
    //  Evolution step
    /////////////////////////////////

    // dV = dt*a^3*dV/df for the field being evolved

    //  Different coefficients related to equations of motion
    //  and to the potential function used
    //  f_coeff[0] = a(t)
    //  f_coeff[1] = dt*a(t)/(dx^2)

    {% if tmp_c %} {{ tmp_terms_c }} {% endif %}

        pi{{ field_i_c }}_m[in_idx] += f_coeff[0]*(D2f - ({{ dV_c }}));
        // pi{{ field_i_c }}_m[in_idx] = D2f;

    //  Note that -4*V_interaction included for the last field
        sumi = f{{ field_i_c }}*D2f {{ V_c }};


    {% if gw_c %}
        //Tensor perturbation source terms:
        //Dxf = smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1];
        //Dyf = smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x];
        //Dzf = sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x];

    //  f_coeff[2] = dt*2*mpl^-2*0.25*a(t)^2/(dx^2)

        piu11_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1]);
        piu12_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
        piu22_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
        piu13_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
        piu23_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
        piu33_m[in_idx] += f_coeff[2]*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
    {% endif %}


     }

    {% set foo = DIM_Z_c-1 %}
    {% set foo2 = DIM_Z_c-2 %}
    /////////////////////////////////////////////////
    // advance in z direction until z={{ foo }}
    // z = {{ foo }} calculated seperately

    volatile unsigned int i;
//#pragma unroll {{ foo2 }}
    for(i=1; i<({{ foo }}); i++)
    {
        in_idx  += {{ stride_c }};

        __syncthreads();

        //////////////////////////////////////////////////////////////
        // Update the inner up, middle and down shared memory blocks
        // by copying middle->down, up->middle and new data into up

	sdwn_data[threadIdx.y][threadIdx.x] = smid_data[threadIdx.y][threadIdx.x];
	smid_data[threadIdx.y][threadIdx.x] = sup_data[threadIdx.y][threadIdx.x];
	sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[in_idx+{{ stride_c }}];
       
	__syncthreads();

	///////////////////////////////////////////////////////////////////
	// Calculate values only for the inner points of the thread block

	if(((threadIdx.x>0)&&(threadIdx.x<({{ blockx }})))&&((threadIdx.y>0)&&(threadIdx.y<({{ blocky }}))))
	{

            f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
            // other fields
           {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
           {% endfor %}

	/////////////////////////////////////////
	// Calculations

        // Discretized Laplacian operator
        // f_coeff[1] = dt*a(t)/(dx^2)
        // c_coeff's = laplacian discretization coefficients

          D2f  = f_coeff[1]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
              + c_coeff[1]*( sdwn_data[threadIdx.y][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x+1]
              + smid_data[threadIdx.y-1][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x] + sup_data[threadIdx.y][threadIdx.x])
              + c_coeff[2]*(sdwn_data[threadIdx.y][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x] 
              + sdwn_data[threadIdx.y][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x]
              + smid_data[threadIdx.y-1][threadIdx.x+1] + smid_data[threadIdx.y-1][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x-1] + smid_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x]
              + sup_data[threadIdx.y][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x] )
              + c_coeff[3]*( sdwn_data[threadIdx.y-1][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x-1]
              + sdwn_data[threadIdx.y+1][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y-1][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x-1]
              + sup_data[threadIdx.y+1][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x+1]));


	/////////////////////////////////
	//  Evolution step
	/////////////////////////////////

        // dV = dt*a^3*dV/df for the field being evolved

        //  Different coefficients related to equations of motion
        //  and to the potential function used
        //  f_coeff[0] = a(t)
        //  f_coeff[1] = dt*a(t)/(dx^2)

    {% if tmp_c %} {{ tmp_terms_c }} {% endif %}

          pi{{ field_i_c }}_m[in_idx] += f_coeff[0]*(D2f - ({{ dV_c }}));
          //pi{{ field_i_c }}_m[in_idx] = D2f;

        //  Note that -4*V_interaction included for the last field
          sumi += f{{ field_i_c }}*D2f {{ V_c }};


        {% if gw_c %}
        //Tensor perturbation source terms:
        //Dxf = smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1];
        //Dyf = smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x];
        //Dzf = sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x];

        //  f_coeff[2] = dt*2*mpl^-2*0.25*a(t)^2/(dx^2)

          piu11_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1]);
          piu12_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
          piu22_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
          piu13_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
          piu23_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
          piu33_m[in_idx] += f_coeff[2]*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
    {% endif %}



         }

     }

    //////////////////////////////////////////
    // The upper most slice of the lattice

    in_idx  += {{ stride_c }};

    __syncthreads();

    // Load the down, middle and up data to shared memory
    // up data now from the bottom of the lattice
    sdwn_data[threadIdx.y][threadIdx.x]  = smid_data[threadIdx.y][threadIdx.x];
    smid_data[threadIdx.y][threadIdx.x]  = sup_data[threadIdx.y][threadIdx.x];
    sup_data[threadIdx.y][threadIdx.x]  = field{{ field_i_c }}[period(blockIdx.y*(blockDim.y-2) + threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2) + threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }})];

    __syncthreads();

    if(((threadIdx.x>0)&&(threadIdx.x<({{ blockx }})))&&((threadIdx.y>0)&&(threadIdx.y<({{ blocky }}))))
    {

        f{{ field_i_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
        // other fields
       {% for i in other_i_c %} f{{i}} = field{{i}}[in_idx];
       {% endfor %}

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator

        D2f  = f_coeff[1]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
              + c_coeff[1]*( sdwn_data[threadIdx.y][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x+1]
              + smid_data[threadIdx.y-1][threadIdx.x] + smid_data[threadIdx.y][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x] + sup_data[threadIdx.y][threadIdx.x])
              + c_coeff[2]*(sdwn_data[threadIdx.y][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x] 
              + sdwn_data[threadIdx.y][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x]
              + smid_data[threadIdx.y-1][threadIdx.x+1] + smid_data[threadIdx.y-1][threadIdx.x-1]
              + smid_data[threadIdx.y+1][threadIdx.x-1] + smid_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x]
              + sup_data[threadIdx.y][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x] )
              + c_coeff[3]*( sdwn_data[threadIdx.y-1][threadIdx.x+1] + sdwn_data[threadIdx.y-1][threadIdx.x-1]
              + sdwn_data[threadIdx.y+1][threadIdx.x-1] + sdwn_data[threadIdx.y+1][threadIdx.x+1]
              + sup_data[threadIdx.y-1][threadIdx.x+1] + sup_data[threadIdx.y-1][threadIdx.x-1]
              + sup_data[threadIdx.y+1][threadIdx.x-1] + sup_data[threadIdx.y+1][threadIdx.x+1]));


    /////////////////////////////////
    //  Evolution step
    /////////////////////////////////

    // dV = dt*a^3*dV/df for the field being evolved

    //  Different coefficients related to equations of motion
    //  and to the potential function used
    //  f_coeff[0] = a(t)
    //  f_coeff[1] = dt*a(t)/(dx^2)

    {% if tmp_c %} {{ tmp_terms_c }} {% endif %}

        pi{{ field_i_c }}_m[in_idx] += f_coeff[0]*(D2f - ({{ dV_c }}));
      //  pi{{ field_i_c }}_m[in_idx] = D2f;

    //  Note that -4*V_interaction included for the last field
        sumi += f{{ field_i_c }}*D2f {{ V_c }};

    // Use atomic add as a precaution
        atomicAdd(&sumterm_w[period(blockIdx.y*(blockDim.y-2) + threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2) + threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }})], sumi);


    {% if gw_c %}
        //Tensor perturbation source terms:
        //Dxf = smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1];
        //Dyf = smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x];
        //Dzf = sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x];

        //  f_coeff[2] = dt*2*mpl^-2*0.25*a(t)^2/(dx^2)

        piu11_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1]);
        piu12_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
        piu22_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x]);
        piu13_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y][threadIdx.x+1] - smid_data[threadIdx.y][threadIdx.x-1])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
        piu23_m[in_idx] += f_coeff[2]*(smid_data[threadIdx.y+1][threadIdx.x] - smid_data[threadIdx.y-1][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
        piu33_m[in_idx] += f_coeff[2]*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x])*(sup_data[threadIdx.y][threadIdx.x] - sdwn_data[threadIdx.y][threadIdx.x]);
    {% endif %}




    }

}


