__global__ void tranposeKernel( float * input , float * output , int M , int N )
{
    int tx = threadIdx.x ;
    int ty = threadIdx.y ;
    int bx = blockIdx.x ;
    int by = blockIdx.y ;

    int x = bx * blockDim.x + tx ;
    int y = by * blockDim.y + ty ;

    __shared__ float tile[32][32] ;
    if( x < N && y < M )
    {
        tile[ty][ty^tx] = input[y * N + x ] ;
    }
    
    __syncthreads() ;

    int xi = blockDim.y * by + tx ;
    int yi = blockDim.x * bx + ty ;

    if( xi < M && yi < N )
    {
        output[yi * M + xi ] = tile[tx][tx^ty] ; 
    }



}