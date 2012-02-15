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


__constant__ {{ type_name_c }} c_coeff[4];
__constant__ {{ type_name_c }} gw_coeff[1];

////////////////////////////////////////////////////////////////////////////////
// Gravitational wave kernels
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculate the laplacian of one component of the u_ij tensor and evolve the canonical momentum of the tensor field
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelU3_{{ tensor_ij_c }}({{ type_name_c }} *u11_m, {{ type_name_c }} *u12_m, {{ type_name_c }} *u22_m, {{ type_name_c }} *u13_m, {{ type_name_c }} *u23_m, {{ type_name_c }} *u33_m, {{ type_name_c }} *piu11_m, {{ type_name_c }} *piu12_m, {{ type_name_c }} *piu22_m, {{ type_name_c }} *piu13_m, {{ type_name_c }} *piu23_m, {{ type_name_c }} *piu33_m)

// This function calculates \nabla^2 u_{ij} for the u_{ij} components
// and evolves pi_u_{ij}
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
    volatile unsigned int stride = {{ stride_c }};

    {{ type_name_c }} D2u;
    //{{ type_name_c }} sumi=0.;

    /////////////////////////////////////////
    // load the initial data into smem
    // sdwn_data from the top of the lattice
    // due to the periodicity of the lattice

    sdwn_data[threadIdx.y][threadIdx.x] = u{{ tensor_ij_c }}_m[in_idx + ({{ DIM_Z_c }} - 1)*stride];
    smid_data[threadIdx.y][threadIdx.x] = u{{ tensor_ij_c }}_m[in_idx];
    sup_data[threadIdx.y][threadIdx.x]  = u{{ tensor_ij_c }}_m[in_idx + stride];

    __syncthreads();

    //////////////////////////////////////////////////////////
    // Calculate values only for the inner points of the block

    if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
    {
        //f{{ tensor_ij_c }} = smid_data[threadIdx.y][threadIdx.x]; 	// current field value
        //pi{{ tensor_ij_c }} = pi{{ tensor_ij_c }}_m[in_idx];		// field derivative

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator
    // gw_coeff[0] = dt*a(t)^2/(dx^2)
    // c_coeff's = laplacian discretization coefficients

        D2u = gw_coeff[0]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
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


    /////////////////////////////////////////
    //  Evolution step of the momentum term
    /////////////////////////////////////////

    //  Different coefficients related to equations of motion
    //  and to the potential function used
    //  gw_coeff[0] = dt*a(t)^2/(dx^2)

        piu{{ tensor_ij_c }}_m[in_idx] += D2u;

     }

    /////////////////////////////////////////////////
    // advance in z direction until z={{ DIM_Z_c }}-1
    // z = {{ DIM_Z_c }}-1 calculated seperately

    volatile unsigned int i;
    for(i=1; i<({{ DIM_Z_c }}-1); i++)
    {
        in_idx  += stride;

        __syncthreads();

        //////////////////////////////////////////////////////////////
        // Update the inner up, middle and down shared memory blocks
        // by copying middle->down, up->middle and new data into up

	sdwn_data[threadIdx.y][threadIdx.x] = smid_data[threadIdx.y][threadIdx.x];
	smid_data[threadIdx.y][threadIdx.x] = sup_data[threadIdx.y][threadIdx.x];
	sup_data[threadIdx.y][threadIdx.x]  = u{{ tensor_ij_c }}_m[in_idx+stride];
       
	__syncthreads();

	///////////////////////////////////////////////////////////////////
	// Calculate values only for the inner points of the thread block

	if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
	{

	/////////////////////////////////////////
	// Calculations

        // Discretized Laplacian operator
        // gw_coeff[0] = dt*a(t)/(dx^2)
        // c_coeff's = laplacian discretization coefficients

          D2u  = gw_coeff[0]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
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


	////////////////////////////////////////
	//  Evolution step of the momentum term
	////////////////////////////////////////

        //  Different coefficients related to equations of motion
        //  and to the potential function used
        //  gw_coeff[0] = a(t)
        //  gw_coeff[0] = dt*a(t)/(dx^2)


          piu{{ tensor_ij_c }}_m[in_idx] += D2u;

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
    sup_data[threadIdx.y][threadIdx.x]  = u{{ tensor_ij_c }}_m[period(blockIdx.y*(blockDim.y-2) + threadIdx.y-1,blockIdx.y,{{ grid_y_c }},{{ DIM_Y_c }})*{{ DIM_X_c }} + period(blockIdx.x*(blockDim.x-2) + threadIdx.x-1,blockIdx.x,{{ grid_x_c }},{{ DIM_X_c }})];

    __syncthreads();

    if(((threadIdx.x>0)&&(threadIdx.x<({{ block_x_c }}-1)))&&((threadIdx.y>0)&&(threadIdx.y<({{ block_y_c }}-1))))
    {

    /////////////////////////////////////////
    // Calculations

    // Discretized Laplacian operator

        D2u  = gw_coeff[0]*(c_coeff[0]*smid_data[threadIdx.y][threadIdx.x]
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

    /////////////////////////////////////////
    //  Evolution step of the momentum term
    /////////////////////////////////////////

    //  Different coefficients related to equations of motion
    //  and to the potential function used
    //  gw_coeff[0] = a(t)
    //  gw_coeff[0] = dt*a(t)/(dx^2)

        piu{{ tensor_ij_c }}_m[in_idx] += D2u;

    }

}


