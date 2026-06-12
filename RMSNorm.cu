#include <cuda_runtime.h>

// 修正后的求和规约
__device__ float float_Sum_Reduce(float val) {
    const int tx = threadIdx.x;
    int lane = tx % warpSize;
    int wid = tx / warpSize;

    // 1. Warp 内规约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 2. 共享内存存放每个 Warp 的和
    __shared__ float warpSump[32]; 
    if (lane == 0) warpSump[wid] = val;
    __syncthreads();

    // 3. 第一个 Warp 汇总
    int num_wraps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        val = (lane < num_wraps) ? warpSump[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    } else {
        val = 0.0f;
    }
    return val; 
}

__global__ void RmsNorm(float *input, float *output, float *wei, float eps, int batch, int size) {
    const int bx = blockIdx.x;
    if (bx >= batch) return;

    float *in = input + size * bx;
    float *out = output + size * bx;
    float sum = 0.0f;

    // 计算平方和
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float tmp = in[i];
        sum += tmp * tmp;
    }

    // 整个 Block 规约
    sum = float_Sum_Reduce(sum);

    // 关键修复：使用共享内存广播结果给全 Block
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        // 使用 rsqrtf (1/sqrt) 直接计算缩放系数
        s_rms = rsqrtf(sum / static_cast<float>(size) + eps);
    }
    __syncthreads(); // 确保所有线程都拿到了 s_rms

    float rms_scale = s_rms;

    // 应用缩放和权重
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = in[i] * wei[i] * rms_scale;
    }
}



__device__ float float_Sum_Reduce(float val )
{
    const int tx = threadidx.x ;
    int lane = threadIdx.x % warpSize ;
    int wrap_id = threadIdx.x / warpSize ;
    
    for( int offset = warpSize /2 ; offset > 0 ; offset /= 2 )
    {
        val += __shfl_down_sync (0xFFFFFFFF, val , offset) ;
    }
    __shared__ float wrapsum[32] ;
    if( lane == 0 )
    {
        wrapsum[wrap_id] = val ;
    }
    __syncthreads() ;
    int warpnum = (blockDim.x + warpSize - 1) / warpSize ;
    if( wrap_id == 0 ){
        val = (lane<warpnum):wrapsum[lane] ? 0.0f ;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }else{
        val = 0.0f ;
    }



    return val ; 

}
__device__ float blockReduceMax(float val)
{
    const int tx = threadIdx.x ;
    int lane = tx % warpSize ;
    int warp_id = tx / warpSize ;
    for( int offset = warpSize / 2 ; offset > 0 ; offset /= 2)
    {
        val = fmaxf(val,__shfl_down_sync(0xFFFFFFFF,val,offset)) ;
    }
    __shared__ float warpMaxp[32] ;
    if( lane == 0 ) warpMaxp[warp_id] = val ;
    __syncthreads() ;
    int warpNums = ( blockDim.x + warpSize - 1 ) / warpSize ;
    if( warp_id == 0 ){
        val = (lane < warpNums) ? warpMaxp[lane] : -INFINITY ;
        for( int offset = warpSize / 2; offset > 0 ; offset /=2 )
        {
            val = fmaxf(val ,__shfl_down_sync(__activemask(),val,offset)) ;
        }
    }else{
        val = -INFINITY ;
    }


    return val ;
}

__device__ float blockReduceSum(float val)
{
    const int tx = threadIdx.x ;
    int lane = tx % warpSize ;
    int warp_id = tx / warpSize ;
    for( int offset = warpSize / 2 ; offset > 0 ; offset /= 2 )
    {
        val += __shfl_down_sync(__activemask(),val , offset) ;
    }
    __shared__ float warpSump[32] ;
    if( lane == 0 ) warpSump[warp_id] = val ;
    __syncthreads() ;
    int warpNums = (blockDim.x + warpSize - 1 ) / warpSize ; 
    if( warp_id == 0 ){
        val = (lane < warpNums ) ? warpSump[warp_id] : 0.0f ;
        for( int offset = warpSize / 2 ; offset > 0 ; offset /= 2)
        {
            val += __shfl_down_sync(__activemask,val , offset ) ;
        }

    }else{
        val = 0.0f ;
    }

    return val ;

}

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
__device__ float blockReduceSum(float val)
{
    int bx = blockIdx.x ;
    int lane = threadIdx.x % warpSize ;
    int warp_id = threadIdx
    for( int offset = warpSize /2 ; offset > 0 ; offset /= 2)
    {
        val += __shfl_down_sync(__activemask , val , offset) ;
    }
    __shared__ float wrapSump[32] ;
    if( lane == 0) warpSump[warp_id] = val ;
    __syncthreads() ;
    int warpnump =( blockDim.x + warpSize - 1 )/ warpSize ;
    if( warp_id == 0){
        val = (lane < warpnump) ? wrapSump[lane] : 0.0f ;
        for( int offset = warpSize /2 ; offset > 0 ; offset /= 2 )
        {
             val += __shfl_down_sync(__activemask , val , offset) ;
        }
    }else{
        val = 0.0f ;
    }
    return val ;
}

__global__ void RmsNormKernel(float * input , float * output , int batch , int size)
{
    int bx = blockIdx.x ;
    if( bx > batch ) return ;
    float *x = input + bx * size ;
    float *y = output + bx * size ;
    float val = 0.0f ;
    for( int i = threadIdx.x ; i < size ; i += blockDim.x )
    {
        float tmp = x[i] ; 
        val += tmp * tmp  ;
    }
    val = blockReduceSum(val) ;
    __shared__ float rstd ;
    if( threadIdx.x == 0 )
    {
        rstd = rsqrtf(val/static_cast<float>(size) + 1e-6) ;
    }
    __syncthreads() ;
    for( int i = threadIdx.x ; i < size ; i += blockDim.x )
    {
        float tmp = x[i] ; 
        y[i] = tmp * rstd ;
    }
}