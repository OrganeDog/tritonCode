
__global__ void BatchNorm( float * input , float * output , int N , int H , int W , int C )
{
    int cx = blockIdx.x ;
    if( cx >= C) return ;
    int total_elements = N * W * H ;
    float local_sum = 0.0f ;
    float local_sqsum = 0.0f ;
    for( int i = threadIdx.x ; i < total_elements ; i += blockDim.x )
    {
        int global_idx = cx * total_elements + i ;
        float tmp = input[global_idx] ;
        local_sum += tmp ;
        local_sqsum += tmp * tmp ;

    }

    local_sum = blockReduceSum(local_sum) ;
    local_sqsum = blockReduceSum( local_sqsum ) ;
    __shared__ float mean , rstd ;
    if( threadIdx.x == 0 ){
        mean = local_sum / static_cast<float>(total_elements) ;
        rstd = fsqrtf(local_sqsum / static_cast<float>(total_elements) - mean * mean ); 
    }
    __syncthreads() ;
    for( int i = threadIdx.x ; i < total_elements ; i += blockDim.x )
    {
        int global_idx = cx * total_elements + i ;
        output[global_idx] = (input[global_idx] - mean ) / rstd ;
    }

}