////////////////////////////////////////////////////////////////////////////////
// Fourier space calculations
////////////////////////////////////////////////////////////////////////////////
//

__global__ void gpu_k_vec({{ real_name_c  }} *k_x, {{ real_name_c  }} *k_y, {{ real_name_c  }} *k_z, {{ real_name_c  }} *k2_abs)
//
// Calculate k_eff values corresponding to the discrete derivative operator
//
{
    volatile int ix  = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    int iz;
    volatile unsigned int in_idx = (blockIdx.y*blockDim.y+threadIdx.y)*{{ DIM_X }} + (blockIdx.x*blockDim.x+threadIdx.x);

    int stride = {{ stride  }};
    {{ real_name_c  }} ic, jc, kc, is, js, ks;
    {{ real_name_c  }} kx, ky, kz, k2_val;

    ic = cos({{ w_c }}*({{ real_name_c  }})ix);
    jc = cos({{ w_c }}*({{ real_name_c  }})iy);

    is = sin({{ w_c }}*({{ real_name_c  }})ix);
    js = sin({{ w_c }}*({{ real_name_c  }})iy);

    // Move in z-direction in g_output
    for(iz =0; iz < {{ DIMZ2 }}; iz++)
    {
        kc = cos({{ w_c }}*({{ real_name_c  }})iz);
        ks = sin({{ w_c }}*({{ real_name_c  }})iz);

	{% if method == "defrost"%}
          
            kx = {{ dk }}*(is);
            ky = {{ dk }}*(js);
            kz = {{ dk }}*(ks);

            k2_val = {{ dk2 }}*(-({{ ct_0 }}) - {{ ct_1 }}*(ic+jc+kc) - {{ ct_2 }}*(ic*jc+ic*kc+jc*kc) - {{ ct_3 }}*(ic*jc*kc));
	
	{% elif method == "hlattice" and radius_c == 2 %}

            kx = {{ dk }}*(is);
            ky = {{ dk }}*(js);
            kz = {{ dk }}*(ks);

            k2_val = kx*kx + ky*ky + kz*kz;
	
        {% elif method == "hlattice" and radius_c == 4 %}

            kx = 0.3333333333333333*{{ dk }}*is*(4.0 - ic);
            ky = 0.3333333333333333*{{ dk }}*js*(4.0 - jc);
            kz = 0.3333333333333333*{{ dk }}*ks*(4.0 - kc);

            k2_val = kx*kx + ky*ky + kz*kz;

        {% elif method == "std" %}

            kx = 0.3333333333333333*{{ dk }}*is*(4.0 - ic);
            ky = 0.3333333333333333*{{ dk }}*js*(4.0 - jc);
            kz = 0.3333333333333333*{{ dk }}*ks*(4.0 - kc);

            k2_val = kx*kx + ky*ky + kz*kz;

        {% endif %}

        if (k2_val < 0.0) {

            k_x[in_idx] = 0.0;
            k_y[in_idx] = 0.0;
            k_z[in_idx] = 0.0;

            k2_abs[in_idx] = 0.0;
        }
        else {
            k_x[in_idx] = kx;
            k_y[in_idx] = ky;
            k_z[in_idx] = kz;

            k2_abs[in_idx] = k2_val;
        }

        in_idx += stride;
    }

}



