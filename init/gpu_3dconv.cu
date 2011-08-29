////////////////////////////////////////////////////////////////////////////////
// Initialize 3D convolution kernel (using linear interpolation of radial profile)
////////////////////////////////////////////////////////////////////////////////
// See DEFROST paper for explanation

__global__ void gpu_3dconv(double *g_output,
                            double *g_input,
                            int nn,
                            double osf,
                            int dimx,
                            int dimy,
                            int dimz )
{
    volatile int ix  = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    int iz;
    int l;
    int stride = dimx*dimy;
    double g0 = g_input[0];
    double g1 = g_input[1];
    double kk;

    // Move in z-direction in g_output
    for(iz =0; iz < dimz; iz++)
    {
        kk = sqrt(int2double((ix+1-nn)*(ix+1-nn) + (iy+1-nn)*(iy+1-nn) + (iz+1-nn)*(iz+1-nn)))*osf;

        l = double2int(floor(kk));

        if (l > 0) {
            g_output[iz*stride + iy*dimx + ix] = g_input[l-1] + (kk - int2double(l))*(g_input[l] - g_input[l-1]);
        }
        else {
            g_output[iz*stride + iy*dimx + ix] = (4.0*g0-g1)/3.0;
        }
    }

}

