/**
 * Most simple example. Launch a kernel that does nothing on the device.
 * To compile: nvcc -o 01-mostbasic 01-mostbasic.cu
 */


/**
 * The __global__ keyword indicates that the function (kernel) 
 * runs on the device and is callable from the host.
 */
__global__ void donothing()
{}


int main()
{
  // Call the kernel on the CUDA device using 1 block of 1 thread.
  donothing<<< 1, 1 >>>();
  
  return 0;
}
