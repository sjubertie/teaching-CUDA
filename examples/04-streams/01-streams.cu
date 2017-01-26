/*
 * Streams.
 * Simple example to show how to overlap computation and communication.
 * In this case, the computation is negligible compared to the communication but
 * using the Nvidia visual profiler shows that communication from the host to the
 * device and from the device to the host may overlap.
 */
#include <iostream>
#include <vector>


__global__ void vecadd( int * v0, int * v1, std::size_t size )
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid < size )
  {
    v0[ tid ] += v1[ tid ];
  }
}


int main()
{
  std::size_t const size = 1000000;
  std::size_t const sizeb = size * sizeof( int );

  int * v0 = nullptr;
  int * v1 = nullptr;
	
  // Allocation on the host is done with the cudaMallocHost function.
  // It is mandatory for streams since the memory needs to be pinned
  // i.e. fixed in RAM and not swapable.
  cudaMallocHost( &v0, sizeb );
  cudaMallocHost( &v1, sizeb);
  
  for( std::size_t i = 0 ; i < size ; ++i )
  {
    v0[ i ] = v1[ i ] = i;
  }
  
  int * v0_d = nullptr;
  int * v1_d = nullptr;

  cudaMalloc( &v0_d, sizeb );
  cudaMalloc( &v1_d, sizeb );
  
  // Streams declaration.
  cudaStream_t streams[ 2 ];

  // Creation.
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );
  
  // Input vectors are send by halves.
  // The cudaMemcpyAsync function is used instead of the usual cudaMemcpy function
  // since it takes the stream as its last parameter.
  cudaMemcpyAsync( v0_d, v0, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( v1_d, v1, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 0 ] );
  
  cudaMemcpyAsync( v0_d+size/2, v0+size/2, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 1 ] );
  cudaMemcpyAsync( v1_d+size/2, v1+size/2, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 1 ] );
  
  dim3 block( 1024 );
  dim3 grid( (size - 1) / block.x * block.x + 1 );
  
  // One kernel is launched in each stream.
  vecadd<<< 1, size/2, 0, streams[ 0 ] >>>( v0_d, v1_d, size/2 );

  vecadd<<< 1, size/2, 0, streams[ 1 ] >>>( v0_d+size/2, v1_d+size/2, size/2 );
 
  // Sending back the resulting vector by halves.
  cudaMemcpyAsync( v0, v0_d, size/2 * sizeof(int), cudaMemcpyDeviceToHost, streams[ 0 ] );
  cudaMemcpyAsync( v0+size/2, v0_d+size/2, size/2 * sizeof(int), cudaMemcpyDeviceToHost, streams[ 1 ] );
  
  // Synchronize everything.
  cudaDeviceSynchronize();
  
  // Destroy streams.
  cudaStreamDestroy( streams[ 0 ] );
  cudaStreamDestroy( streams[ 1 ] );
     
  for( std::size_t i = 0 ; i < size ; ++i )
  {
    std::cout << v0[ i ] << std::endl;
  }
  
  cudaFree( v0_d );
  cudaFree( v1_d );
 
  cudaFreeHost( v0 );
  cudaFreeHost( v1 );
 
  return 0;
}
