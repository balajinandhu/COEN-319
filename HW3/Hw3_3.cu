#include<stdio.h>
#include<cuda.h>
#include<vector_types.h>

#define TILE_WIDTH 72


void matMulSeq(float *A, float *B, float *C, int numARows, int numAColumns,
			int numBRows, int numBColumns, int numCRows, int numCColumns){
	int Row, Col, k;
	float temp;
	for(Row=0; Row<numARows; Row++){
		for(Col=0; Col<numBColumns; Col++){
			 temp = 0.0;
			for(k=0; k<numBRows; k++){
				temp+=A[Row*numAColumns+k]+B[k*numBColumns+Col];
			}
		C[Row*numCColumns+Col] = temp;
		}
	}
}


__global__ void matMul(float *A, float *B, float *C, int numARows, int numAColumns,
			int numBRows, int numBColumns, int numCRows, int numCColumns) {


	//do matrix mult

__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x, by = blockIdx.y,
	tx = threadIdx.x, ty = threadIdx.y,
	Row = by*TILE_WIDTH + ty,
	Col = bx*TILE_WIDTH + tx;

float Pvalue = 0;
for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

int main(int argc, char ** argv) {
    //wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * hostCSeq;
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows = 1000; // number of rows in the matrix A
    int numAColumns = 10000; // number of columns in the matrix A
    int numBRows = 10000; // number of rows in the matrix B
    int numBColumns = 1000; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    hostA = (float*) malloc(numARows*numAColumns * sizeof(float));
    hostB = (float *) malloc(numBRows*numBColumns * sizeof(float));
 
    static int m = 1;
    for (int i = 0; i < numARows; i++)
    {
	srand(7*m+time(0));
	for (int k = 0; k < numAColumns; k++) 
	{
//		hostA[i*numAColumns+k]=1;
	    hostA[i*numAColumns+k] = rand() % 100 + 1; 
	}
	m++;
    }
     for (int i = 0; i < numBRows; i++)
    {
	srand(7*m+time(0));
	for (int k = 0; k < numBColumns; k++) 
	{
//		hostB[i*numBColumns+k] = 1;
	    hostB[i*numBColumns+k] = rand() % 100 + 1; 
	}
	m++;
    }
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
    hostCSeq = (float *)malloc(sizeof(float) * numCRows * numCColumns);


    //@@ Allocate GPU memory here
    
    cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns);
    cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns);
    cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns);


    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numCColumns-1)/TILE_WIDTH+1, (numCRows-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    float elapsedTime;
  cudaEventRecord(start, 0);

    matMulSeq(hostA, hostB, hostCSeq, numARows, numAColumns,
					numBRows, numBColumns,
					numCRows, numCColumns); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
 
    printf("\nElapsed time for seq: %f ms.", elapsedTime);
    //@@ Launch the GPU Kernel here
    cudaEventRecord(start, 0);

        matMul<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);
   
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
//@@ Compare the sequential and parallel output
 for (int i = 0; i < numCRows; i++)
    {
	for (int k = 0; k < numCColumns; k++) 
	{
	if(hostC[i*numCColumns+k]-hostCSeq[i*numCColumns+k]==0.0f)
		printf("No match");
	}
    }
    printf("\nElapsed time for parallel: %f ms.", elapsedTime);
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //wbTime_stop(GPU, "Freeing GPU Memory");

    //wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
