template < const int BLOCK_SIZE >
__global__ gemm_v1( float * A , float * B ,float * C , int M , int K , int N , float alpha ,float beta )
{
    const int BM = BLOCK_SIZE ;
    const int BN = BLOCK_SIZE ;
    const int BK = BLOCK_SIZE ;

    int bx = blockIdx.x ;
    int by = blockIdx.y ;
    int tx = threadIdx.x % BN ;
    int ty = threadIdx.x / BN ;

    A = &A[by * BM * K ] ;
    B = &B[bx * BN ] ;
    C = &C[by * BM * N + bx * BN ] ;
    __shared__ float As[BM*BK] ;
    __shared__ float Bs[BK*BN] ;
    float tmp = 0.0f ;
    for( int k = 0 ; k < K ; k += BK )
    {
        As[ty * BK + tx] = A[ty * K + tx ] ;
        Bs[ty * BN + tx] = B[ty * N + tx ] ; // 线程块所有的线程共用
        __syncthreads() ;
        A += BK ;
        B += BK * N ;
        for( int i = 0 ; i < BK ; i++ )
        {
            tmp += As[ty * BK + i ] * Bs[i * BN + tx ];

        }
        __syncthreads() ;
    } 
    C[ty * N + tx ] = tmp * alpha + beta * C[ty * N + tx] ;
}
template < const int BLOCK_SIZE >
__global__ gemm_v1( float * A , float * B ,float * C , int M , int K , int N , float alpha ,float beta )
{
    const int BM ;
    const int BN ;
    const int BK ;
    int bx = blockIdx.x ;
    int by = blockIdx.y ;
    





}