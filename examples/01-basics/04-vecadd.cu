/**
 * Add 2 arrays of 100 elements on the device.
 */
#include <iostream>
#include <vector>
#include <algorithm>


__global__ void vecadd( int * v0, int * v1, std::size_t size )
{
  auto tid = threadIdx.x;
  v0[ tid ] += v1[ tid ];
}


int main()
{
  std::vector< int > v0( 100 );
  std::vector< int > v1( 100 );
  
  int * v0_d = nullptr;
  int * v1_d = nullptr;

  for( std::size_t i = 0 ; i < v0.size() ; ++i )
  {
    v0[ i ] = v1[ i ] = i;
  }
  
  cudaMalloc( &v0_d, v0.size() * sizeof( int ) );
  cudaMalloc( &v1_d, v1.size() * sizeof( int ) );

  cudaMemcpy( v0_d, v0.data(), v0.size() * sizeof( int ), cudaMemcpyHostToDevice );
  cudaMemcpy( v1_d, v1.data(), v1.size() * sizeof( int ), cudaMemcpyHostToDevice );
  
  vecadd<<< 1, 100 >>>( v0_d, v1_d, v0.size() );

  cudaMemcpy( v0.data(), v0_d, v0.size() * sizeof( int ), cudaMemcpyDeviceToHost );

  for( auto const x: v0 )
  {
    std::cout << x << std::endl;
  }

  cudaFree( v0_d );
  cudaFree( v1_d );

  return 0;
}