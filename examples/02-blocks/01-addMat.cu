#include <iostream>
#include <vector>


__global__ void addMat( float * mA_d, float * mB_d, std::size_t w, std::size_t h )
{
  auto x = blockDim.x * blockIdx.x + threadIdx.x;
  auto y = blockDim.y * blockIdx.y + threadIdx.y;

  if( x < w && y < h )
  {
    mA_d[ y * w + x ] += mB_d[ y * w + x ];
  }
}


int main()
{
  size_t const w = 100;
  size_t const h = 100;
  size_t const size = w * h;
  
  std::vector< float > mA ( size );
  std::vector< float > mB ( size );
  std::vector< float > mC ( size );

  float * mA_d = nullptr;
  float * mB_d = nullptr;
  
  std::fill( begin( mA), end( mA ), 1.0f );
  std::fill( begin( mB), end( mB ), 1.0f );

  cudaMalloc( &mA_d, size * sizeof( float ) );
  cudaMalloc( &mB_d, size * sizeof( float ) );

  cudaMemcpy( mA_d, mA.data(), size * sizeof( float ), cudaMemcpyHostToDevice );
  cudaMemcpy( mB_d, mB.data(), size * sizeof( float ), cudaMemcpyHostToDevice );

  dim3 block( 32, 32 );
  dim3 grid( ( w - 1 ) / block.x + 1, ( h - 1 ) / block.y + 1 );
 
 	std::cout << grid.x << ' ' << grid.y << std::endl;
  
  addMat<<< grid, block >>>( mA_d, mB_d, w, h );

  cudaMemcpy( mC.data(), mA_d, size * sizeof( float ), cudaMemcpyDeviceToHost );

  for( std::size_t j = 0 ; j < h ; ++j )
  {
    for( std::size_t i = 0 ; i < w ; ++i )
    {
      std::cout << mC[ j * w + i ] << ' ';
    }
    std::cout << std::endl;
  }
  
  return 0;
}
