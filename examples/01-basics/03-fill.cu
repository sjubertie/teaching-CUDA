/**
 * Fill a vector of 100 ints on the GPU with consecutive values.
 */
#include <iostream>
#include <vector>


__global__ void fill( int * v, std::size_t size )
{
  // Get the id of the thread ( 0 -> 99 ).
  auto tid = threadIdx.x;
  // Each thread fills a single element of the array. 
  v[ tid ] = tid;
}


int main()
{
  std::vector< int > v( 100 );

  int * v_d = nullptr;

  // Allocate an array an the device.
  cudaMalloc( &v_d, v.size() * sizeof( int ) );

  // Launch one block of 100 threads on the device.
  // In this block, threads are numbered from 0 to 99.
  fill<<< 1, 100 >>>( v_d, v.size() );

  // Copy data from the device memory to the host memory.
  cudaMemcpy( v.data(), v_d, v.size() * sizeof( int ), cudaMemcpyDeviceToHost );

  for( auto x: v )
  {
    std::cout << x << std::endl;
  }

  cudaFree( v_d );

  return 0;
}