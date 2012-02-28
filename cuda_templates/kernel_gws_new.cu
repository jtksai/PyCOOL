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
http://www.science.smith.edu/departments/Physics/fstaff/gfelder/latticeeasy/ ,
DEFROST http://www.sfu.ca/physics/cosmology/defrost .
(See http://arxiv.org/abs/0809.4904 for more information.),
Nvidia SDK FDTD3dGPU kernel
(See http://developer.nvidia.com/gpu-computing-sdk .) 
and from HLattice HLattice.
(See http://arxiv.org/abs/1102.0227 for more information.)
*/


__constant__ {{ type_name_c }} c2_coeff[5];
__constant__ {{ type_name_c }} gw_coeff[1];

////////////////////////////////////////////////////////////////////////////////
// Gravitational wave kernels
////////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculate the laplacian of one component of the u_ij tensor and evolve the canonical momentum of the tensor field
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelU3_{{ tensor_ij_c }}({{ type_name_c }} *u11_m, {{ type_name_c }} *u12_m, {{ type_name_c }} *u22_m, {{ type_name_c }} *u13_m, {{ type_name_c }} *u23_m, {{ type_name_c }} *u33_m, {{ type_name_c }} *piu11_m, {{ type_name_c }} *piu12_m, {{ type_name_c }} *piu22_m, {{ type_name_c }} *piu13_m, {{ type_name_c }} *piu23_m, {{ type_name_c }} *piu33_m)

// This function calculates \nabla^2 u_{ij} for the u_{ij} components
// and evolves pi_u_{ij}
{

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

    //volatile unsigned int stride_z = {{ stride_c }};

    //{{ type_name_c }} f{{ field_i_c }};
    //{{ type_name_c }} pi{{ field_i_c }};

    {{ type_name_c }} up[{{ radius_c }}];
    {{ type_name_c }} down[{{ radius_c }}];

   {% for i in other_i_c %} {{ type_name_c }} f{{i}};
   {% endfor %}
    {{ type_name_c }} D2u;

    /////////////////////////////////////////
    // load the initial data into shared mem
    // down data from the top of the lattice
    // due to the periodicity of the lattice

    // Down data
    // In a multi-gpu implementation these values could be loaded from a different device
//#pragma unroll {{ radiusp1 }}
    for (j = 0; j < {{ radius_c }}; j++)
    {
        down[j] = u{{ tensor_ij_c }}_m[in_idx];
        in_idx -= {{ stride_c }};
    }

    //  Inner points of shared memory
    s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx];

    // West halo
    if (blockIdx.x == 0){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx + ({{ blockIdx01 }})];
        }
    }
    else {
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx02 }})];
        }
    }
    // East halo
    if (blockIdx.x == {{ gridx1 }}){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.x < {{ radius_c }}){
            //sumterm_w[out_idx] = out_idx -({{ blockIdx11 }});
            s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx11 }})];
        }
    }
    else {
        if (threadIdx.x < {{ radius_c }}){
            s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdx12 }}];
        }
    }
    // North halo
    if (blockIdx.y == 0){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded from a different device
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy01 }}];
        }
    }
    else {
        if (threadIdx.y < {{ radius_c }}){
            //sumterm_w[out_idx] = out_idx - ({{ blockIdy02 }});
            s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy02 }})];
         }
    }

    // South halo
    if (blockIdx.y == {{ gridy1 }}){
        // Periodic lattice
        // In a multi-gpu implementation these values could be loaded fjennifer lopez heightrom a different device
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy11 }})];
        }
    }
    else {
        if (threadIdx.y < {{ radius_c }}){
            s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy12 }}];
         }
    }

    // Up data
    // In a multi-gpu implementation these values could be loaded from a different device
    in_idx = out_idx + {{ stride_c }};

//#pragma unroll {{ radius_c }}
    for (j = 0 ; j < {{ radius_c }} ; j++)
    {
        up[j] = u{{ tensor_ij_c }}_m[in_idx];
        in_idx += {{ stride_c }};
    }
    
    __syncthreads();

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator
    // gw_coeff[0] = dt*a(t)^2/(dx^2)
    // c2_coeff's = laplacian discretization coefficients

    {% if radius_c == 4%}
        D2u = gw_coeff[0]*(c2_coeff[4]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp4 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm4 }}] +
                                       s_data[threadIdx.y + {{ radiusp4 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm4 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm1 }}] + down[{{ radiusm1 }}]) +
                          c2_coeff[3]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp3 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm3 }}] +
                                       s_data[threadIdx.y + {{ radiusp3 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm3 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm2 }}] + down[{{ radiusm2 }}]) +   
                          c2_coeff[2]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm2 }}] +
                                       s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm2 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm3 }}] + down[{{ radiusm3 }}]) +
                          c2_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm1 }}] +
                                       s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm1 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm4 }}] + down[{{ radiusm4 }}]) +
                          c2_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}]));

    {% endif %}

    /////////////////////////////////
    //  Evolution step
    /////////////////////////////////

        piu{{ tensor_ij_c }}_m[out_idx] += D2u;


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

//#pragma unroll {{ radiusm1 }}
        for (int j = 0 ; j < {{ radiusm1 }}; j++)
            up[j] = up[j + 1];
        up[{{ radiusm1 }}] = u{{ tensor_ij_c }}_m[in_idx];

        in_idx += {{ stride_c }};
        out_idx += {{ stride_c }};

        // West halo
        if (blockIdx.x == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx + ({{ blockIdx01 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx02 }})];
            }
        }
        // East halo
        if (blockIdx.x == {{ gridx1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx11 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdx12 }}];
            }
        }
        // North halo
        if (blockIdx.y == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy01 }}];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy02 }})];
             }
        }
        // South halo
        if (blockIdx.y == {{ gridy1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy11 }})];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy12 }}];
             }
        }
       
	__syncthreads();

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator
    // gw_coeff[0] = dt*a(t)^2/(dx^2)
    // c2_coeff's = laplacian discretization coefficients

    {% if radius_c == 4%}
        D2u = gw_coeff[0]*(c2_coeff[4]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp4 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm4 }}] +
                                       s_data[threadIdx.y + {{ radiusp4 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm4 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm1 }}] + down[{{ radiusm1 }}]) +
                          c2_coeff[3]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp3 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm3 }}] +
                                       s_data[threadIdx.y + {{ radiusp3 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm3 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm2 }}] + down[{{ radiusm2 }}]) +   
                          c2_coeff[2]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm2 }}] +
                                       s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm2 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm3 }}] + down[{{ radiusm3 }}]) +
                          c2_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm1 }}] +
                                       s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm1 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm4 }}] + down[{{ radiusm4 }}]) +
                          c2_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}]));
    {% endif %}

	/////////////////////////////////
	//  Evolution step
	/////////////////////////////////

          piu{{ tensor_ij_c }}_m[out_idx] += D2u;

    }

    {% set foo = DIM_Z_c-radius_c %}
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // up data now from the bottom of the lattice due to periodicity
    //

    in_idx = {{ DIM_X_c }}*(blockIdx.y*blockDim.y + threadIdx.y) + blockIdx.x*blockDim.x + threadIdx.x;

#pragma unroll {{ radius_c }}
    for(i={{ foo }}; i<({{ DIM_Z_c }}); i++)
    {
        __syncthreads();

        // Advance the slice (move the thread-front)
#pragma unroll {{ radiusm1 }}
        for (j = {{ radiusm1 }} ; j > 0 ; j--)
            down[j] = down[j - 1];

        down[0] = s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}];
        s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}] = up[0];


#pragma unroll {{ radius_c }}
        for (j = 0 ; j < {{ radiusm1 }}; j++)
            up[j] = up[j + 1];
        up[{{ radiusm1 }}] = u{{ tensor_ij_c }}_m[in_idx];

        in_idx  += {{ stride_c }};
        out_idx  += {{ stride_c }};



        // West halo
        if (blockIdx.x == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx + ({{ blockIdx01 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx02 }})];
            }
        }
        // East halo
        if (blockIdx.x == {{ gridx1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }} ][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdx11 }})];
            }
        }
        else {
            if (threadIdx.x < {{ radius_c }}){
                s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ blockx1 }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdx12 }}];
            }
        }
        // North halo
        if (blockIdx.y == 0){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy01 }}];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy02 }})];
             }
        }
        // South halo
        if (blockIdx.y == {{ gridy1 }}){
            // Periodic lattice
            // In a multi-gpu implementation these values could be loaded from a different device
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx - ({{ blockIdy11 }})];
            }
        }
        else {
            if (threadIdx.y < {{ radius_c }}){
                s_data[threadIdx.y + {{ blocky1 }}][threadIdx.x + {{ radius_c }}] = u{{ tensor_ij_c }}_m[out_idx + {{ blockIdy12 }}];
             }
        }
       
	__syncthreads();


    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator
    // gw_coeff[0] = dt*a(t)^2/(dx^2)
    // c2_coeff's = laplacian discretization coefficients

    {% if radius_c == 4%}
        D2u = gw_coeff[0]*(c2_coeff[4]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp4 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm4 }}] +
                                       s_data[threadIdx.y + {{ radiusp4 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm4 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm1 }}] + down[{{ radiusm1 }}]) +
                          c2_coeff[3]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp3 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm3 }}] +
                                       s_data[threadIdx.y + {{ radiusp3 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm3 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm2 }}] + down[{{ radiusm2 }}]) +   
                          c2_coeff[2]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp2 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm2 }}] +
                                       s_data[threadIdx.y + {{ radiusp2 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm2 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm3 }}] + down[{{ radiusm3 }}]) +
                          c2_coeff[1]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusp1 }}] +
                                       s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radiusm1 }}] +
                                       s_data[threadIdx.y + {{ radiusp1 }}][threadIdx.x + {{ radius_c }}] +
                                       s_data[threadIdx.y + {{ radiusm1 }}][threadIdx.x + {{ radius_c }}] +
                                       up[{{ radiusm4 }}] + down[{{ radiusm4 }}]) +
                          c2_coeff[0]*(s_data[threadIdx.y + {{ radius_c }}][threadIdx.x + {{ radius_c }}]));
    {% endif %}

	/////////////////////////////////
	//  Evolution step
	/////////////////////////////////

          piu{{ tensor_ij_c }}_m[out_idx] += D2u;


    }

}


