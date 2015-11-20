#include<stdio.h>
#include<cuda.h>
#define BLOCK_SIZE 256 //@@ You can change this

void segscan_seq(float *input, float *output, int len){
 output[0]=0.0f;
 for(int i=1;i<len;i++){
   output[i] = input[i-1]+output[i-1];
  }
}

__global__ void helper(float *input_array, float *aux_array, int size) {
    unsigned int t = threadIdx.x, beg = 2 * blockIdx.x * BLOCK_SIZE;
    if (blockIdx.x) {
       if (beg + t < size)
          input_array[beg + t] += aux_array[blockIdx.x - 1];
       if (beg + BLOCK_SIZE + t < size)
          input_array[beg + BLOCK_SIZE + t] += aux_array[blockIdx.x - 1];
    }
}

__global__ void segscan_parallel(float * input_array, float * scanned_array, float *aux_array, int n) {
    // Load a segment of the input_array vector into shared memory
    __shared__ float block_array[BLOCK_SIZE << 1];
    unsigned int t = threadIdx.x, beg = 2 * blockIdx.x * BLOCK_SIZE;
    if (beg + t < n)
       block_array[t] = input_array[beg + t];
    else
       block_array[t] = 0.0f;
    if (beg + BLOCK_SIZE + t < n)
       block_array[BLOCK_SIZE + t] = input_array[beg + BLOCK_SIZE + t];
    else
       block_array[BLOCK_SIZE + t] = 0.0f;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
       int offset = (t + 1) * stride * 2 - 1;
       if (offset < 2 * BLOCK_SIZE)
          block_array[offset] += block_array[offset - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
       int offset = (t + 1) * stride * 2 - 1;
       if (offset + stride < 2 * BLOCK_SIZE)
          block_array[offset + stride] += block_array[offset];
       __syncthreads();
    }

    if (beg + t < n)
       scanned_array[beg + t] = block_array[t];
    if (beg + BLOCK_SIZE + t < n)
       scanned_array[beg + BLOCK_SIZE + t] = block_array[BLOCK_SIZE + t];
    if (aux_array && t == 0)
       aux_array[blockIdx.x] = block_array[2 * BLOCK_SIZE - 1];
}

int main() {
    float *host_in; 
    float *host_out; // The output array of parallel
    float *dev_in;
    float *dev_out;
    float *dev_aux_array, *dev_aux_scanned_arr; 
    float *out_seq; // The output of sequential code 
    int numElements = 100000; // number of elements in the list
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    host_in = (float*) malloc(numElements * sizeof(float));
    out_seq = (float*) malloc(numElements * sizeof(float));
    srand(time(0)); 
    for (int i = 0; i < numElements; i++)
    {
        host_in[i] = (rand() % 100 + 1);
    }

    cudaHostAlloc(&host_out, numElements * sizeof(float), cudaHostAllocDefault);

    cudaMalloc((void**)&dev_in, numElements*sizeof(float));
    cudaMalloc((void**)&dev_out, numElements*sizeof(float));

    cudaMalloc(&dev_aux_array, (BLOCK_SIZE << 1) * sizeof(float));
    cudaMalloc(&dev_aux_scanned_arr, (BLOCK_SIZE << 1) * sizeof(float));


    cudaMemset(dev_out, 0, numElements*sizeof(float));

    cudaMemcpy(dev_in, host_in, numElements*sizeof(float), cudaMemcpyHostToDevice);
    
    float elapsedTime;
    cudaEventRecord(start, 0);
    
    segscan_seq(host_in, out_seq, numElements);     

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("\nElapsed time for seq: %f ms.", elapsedTime);

    int num_blocks = ceil((float)numElements/(BLOCK_SIZE<<1));
    dim3 dim_block(BLOCK_SIZE, 1, 1);
    dim3 dim_grid(num_blocks, 1, 1);
    
    cudaEventRecord(start, 0);
    
    segscan_parallel<<<dim_grid, dim_block>>>(dev_in, dev_out, dev_aux_array, numElements);
    cudaDeviceSynchronize();
    segscan_parallel<<<dim3(1,1,1), dim_block>>>(dev_aux_array, dev_aux_scanned_arr, NULL, BLOCK_SIZE << 1);
    cudaDeviceSynchronize();
    helper<<<dim_grid, dim_block, 0>>>(dev_out, dev_aux_scanned_arr, numElements);
    cudaDeviceSynchronize();
    
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nElapsed time for parallel: %f ms.", elapsedTime);
    cudaMemcpy(host_out, dev_out, numElements*sizeof(float), cudaMemcpyDeviceToHost);
    
    memmove(&host_out[1], &host_out[0], (numElements-1)*sizeof(float));
    host_out[0] = 0.0f;
/*
    for(int j=0;j<10;j++){
        printf("%f  ", host_out[j]);
	printf("%f", out_seq[j]);
}
	printf("\n");*/
     for (int k = 0; k < numElements; k++)
     {
        if(host_out[k]-out_seq[k]!=0.0f){
               printf("%f", host_out[k]);
               printf("%f", out_seq[k]);
               break;
	}
     } 
    //for(int i=0;i<3;i++)
//	cudaStreamDestroy(streams[i]);	

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_aux_array);
    cudaFree(dev_aux_scanned_arr);
    free(out_seq);
    free(host_in);
    cudaFreeHost(host_out);

    return 0;
}
