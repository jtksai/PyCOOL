////////////////////////////////////////////////////////////////////////////////
// Fourier space calculations
////////////////////////////////////////////////////////////////////////////////
// See also DEFROST paper

__global__ void gpu_k2({{ real_name_c  }} *g_output)
//
// Calculate k^2_eff values corresponding to the discrete nabla squared operator
//
{
    volatile int ix  = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    int iz;
    volatile unsigned int in_idx = (blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X }} + (blockIdx.x*blockDim.x+threadIdx.x);

    int stride = {{ stride  }};
    {{ real_name_c  }} ii, jj, kk, k2_val;

    ii = cos({{ w_c }}*({{ real_name_c  }})ix);
    jj = cos({{ w_c }}*({{ real_name_c  }})iy);

    // Move in z-direction in g_output
    for(iz =0; iz < {{ DIMZ2 }}; iz++)
    {
        kk = cos({{ w_c }}*({{ real_name_c  }})iz);

        k2_val = (-({{ ct_0 }}) - {{ ct_1 }}*(ii+jj+kk) - {{ ct_2 }}*(ii*jj+ii*kk+jj*kk) - {{ ct_3 }}*(ii*jj*kk));

        /*
        */
        if (k2_val < 0.0) {
            g_output[in_idx] = 0.0;
        }
        else {
            g_output[in_idx] = k2_val*({{ dk2 }});
            //g_output[in_idx] = k2_val;
        }
        //g_output[in_idx] = k2_val;

        in_idx += stride;
    }

}



__global__ void gpu_k2_to_bin({{ real_name_c  }} *k2_array, {{ real_name_c  }} *k2_bins, int *k2_bin_id, int bins)
//
// Calculate the k2 bin into which a value of k^2 calculated above belongs to
//
{
    volatile unsigned int in_idx = (blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X }} + (blockIdx.x*blockDim.x+threadIdx.x);

    int stride = {{ stride  }};
    int iz, bin;
    {{ real_name_c  }} k2_val, bin_val;

    // Move in z-direction in g_output
    for(iz =0; iz < {{ DIMZ2 }}; iz++)
    {

        k2_val = k2_array[in_idx];

        for(bin =0; bin < bins; bin++)
        {
            bin_val = k2_bins[bin];
            if (fabs(k2_val-bin_val)<1e-14*({{ dk2 }}))
            {
                k2_bin_id[in_idx] = bin;
            }

        }

        in_idx += stride;
    }

}


__global__ void gpu_evolve_lin_fields({{ complex_name_c }} *Fk_1_m{% for i in range(2,fields_c+1) %}, {{ complex_name_c }} *Fk_{{i}}_m{% endfor %}, {{ complex_name_c }} *Pik_1_m{% for i in range(2,fields_c+1) %}, {{ complex_name_c }} *Pik_{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_10_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_10_{{i}}_m{% endfor %}, int *k2_bin_id)
//
// Evolve the perturbation fields by multiplying initial values with proper solutions
//
{
    volatile unsigned int in_idx = (blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X }} + (blockIdx.x*blockDim.x+threadIdx.x);

    int stride = {{ stride  }};
    int iz, k2_bin;
    {{ real_name_c  }} Fk_0, Pik_0;

    // Move in z-direction in g_output
    for(iz =0; iz < {{ DIMZ2 }}; iz++)
    {

        k2_bin = k2_bin_id[in_idx];

        {% for i in range(1,fields_c+1) %}
        Fk_0 = Fk_{{i}}_m[in_idx].x;
        Pik_0 = Pik_{{i}}_m[in_idx].x;

        Fk_{{i}}_m[in_idx].x = Fk_0*f_lin_10_{{i}}_m[k2_bin] + Pik_0*f_lin_01_{{i}}_m[k2_bin];
        Pik_{{i}}_m[in_idx].x = Fk_0*pi_lin_10_{{i}}_m[k2_bin] + Pik_0*pi_lin_01_{{i}}_m[k2_bin];

        Fk_0 = Fk_{{i}}_m[in_idx].y;
        Pik_0 = Pik_{{i}}_m[in_idx].y;

        Fk_{{i}}_m[in_idx].y = Fk_0*f_lin_10_{{i}}_m[k2_bin] + Pik_0*f_lin_01_{{i}}_m[k2_bin];
        Pik_{{i}}_m[in_idx].y = Fk_0*pi_lin_10_{{i}}_m[k2_bin] + Pik_0*pi_lin_01_{{i}}_m[k2_bin];

        {% endfor %}

        in_idx += stride;
    }

}



