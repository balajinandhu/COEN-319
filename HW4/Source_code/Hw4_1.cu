#include<stdio.h>
#include<cuda.h>
#include<vector_types.h>

#define FILTER_WIDTH 5

#define TILE_WIDTH 16
#define BLOCK_WIDTH 32

void convolution_seq(float* inputImage, float *outputImage, float * filterMatrix, int imageWidth, int imageHeight){

        int i,j,s,t;
        //for(i=2; i<imageWidth-2; )
        for(i=0;i<imageHeight;i++){
                for(j=0;j<imageWidth;j++){
                        float temp = 0.0;
                        for(s=0;s<FILTER_WIDTH;s++){
                                for(t=0;t<FILTER_WIDTH;t++){
                                        if((i-2+s>=0) && (j-2+t>=0) && (i-2+s<imageHeight) && (j-2+t<imageWidth))
                                                temp += inputImage[(i-2+s)*imageWidth+(j-2+t)] * filterMatrix[s*FILTER_WIDTH+t];
                                }
                        }
                        outputImage[(i*imageWidth)+j] = temp;
                }
        }

}
__global__ void convolution_kernel(float* deviceInputMatrix, float* deviceOutputMatrix, float* deviceFilter, int width, int height)
{       
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        
        __shared__ float shared_img[BLOCK_WIDTH][BLOCK_WIDTH];
        
        int row_out_index = ty + (blockIdx.y*TILE_WIDTH);
        int col_out_index = tx + (blockIdx.x*TILE_WIDTH);
        
        int row_in_index = row_out_index - 2;
        int col_in_index = col_out_index - 2;
        
        if( (row_in_index>=0) && (col_in_index>=0) && (row_in_index<height) && (col_in_index<width))
        {       
                shared_img[ty][tx] = deviceInputMatrix[row_in_index*width+col_in_index];
        }
        else
        {       
                shared_img[ty][tx] = 0.0f;
        }
        
        __syncthreads();


        
        float output = 0.0f;
        
        int i,j;
        if( ty< TILE_WIDTH && tx<TILE_WIDTH)
        {       
                for(i=0;i<FILTER_WIDTH;i++)
                {       
                        for(j=0;j<FILTER_WIDTH;j++)
                        {
                                output = output + shared_img[ty+i][tx+j]*deviceFilter[i*FILTER_WIDTH+j];

                        }
                }
        }
        __syncthreads();

        if(tx < TILE_WIDTH && ty <TILE_WIDTH && row_out_index < height && col_out_index < width)
        {
                deviceOutputMatrix[(row_out_index*width)+col_out_index] = output;
        }

}

int main(){
    int x= 10000;
    int y=1000;
    float *inputMatrix;
    float *filterMatrix;
    float *outputMatrix;
    float *outputMatrixCuda;
    float * deviceInputMatrix;
    float * deviceOutputMatrix;
    float * deviceFilter;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    inputMatrix = (float*) malloc(x*y * sizeof(float));
    filterMatrix = (float *) malloc(5*5 * sizeof(float));
    outputMatrix = (float*) malloc(x*y * sizeof(float));
    outputMatrixCuda = (float*) malloc(x*y * sizeof(float));

    for (int i = 0; i < x; i++)
    {
        srand(time(0));
        for (int k = 0; k < y; k++)
        {
//          if(i<2||k<2||i>7||k>7)
//              inputMatrix[i*9+k] = 0;
//          else
                inputMatrix[i*y+k] = rand()%100+1;
        }
    }
     for (int i = 0; i < FILTER_WIDTH; i++)
    {
        for (int k = 0; k < FILTER_WIDTH; k++)
        {
            filterMatrix[i*FILTER_WIDTH+k] = 1;
        }
    }
    cudaMalloc((void **) &deviceInputMatrix, y * x * sizeof(float));
    cudaMalloc((void **) &deviceOutputMatrix, y * x * sizeof(float));
    cudaMalloc((void **) &deviceFilter, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    float elapsedTime;
    cudaEventRecord(start, 0);

    convolution_seq(inputMatrix, outputMatrix, filterMatrix, y, x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("\nElapsed time for seq: %f ms.", elapsedTime);

    cudaMemcpy(deviceInputMatrix, inputMatrix, y * x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, filterMatrix, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(BLOCK_WIDTH,BLOCK_WIDTH);
    dim3 grid((y-1)/TILE_WIDTH+1,(x-1)/TILE_WIDTH+1,3);

    cudaEventRecord(start, 0);

    convolution_kernel<<<grid,block>>>(deviceInputMatrix,deviceOutputMatrix,deviceFilter,y,x);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(outputMatrixCuda, deviceOutputMatrix, y * x * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nElapsed time for parallel: %f ms.", elapsedTime);
    cudaFree(deviceInputMatrix);
    cudaFree(deviceOutputMatrix);
    cudaFree(deviceFilter);

    free(filterMatrix);
 /*  for(int i=0;i<x;i++){
        printf("\n");
        for(int j=0;j<y;j++)
            printf("%f", outputMatrixCuda[i*y+j]);
   }*/
//@@ Compare the sequential and parallel output
 for (int i = 0; i < x; i++)
    {
	//printf("\n");
        for (int k = 0; k < y; k++)
        {
        if(outputMatrixCuda[i*y+k]-outputMatrix[i*y+k]!=0.0f){
		printf("%f   ", outputMatrixCuda[i*y+k]);
               printf("No match");
                break;
            }
        }
    }

free(outputMatrix);
return 0;

}

