#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdlib>
#include<string>
#include<iterator>
#include<algorithm>
#include<time.h>
#include<ctime>
#include<omp.h>

using namespace std;
#define NUM_T 1
#define NROWS 1000
#define NCOLUMNS 150

// compile with g++ -std=c++1y -fopenmp


void mmult_seq(int A[][NCOLUMNS], int B[][NROWS], int C[][NROWS]){

	
	for (int i = 0; i < NROWS; i++)
		 for (int j = 0; j < NROWS; j++)
		 {
			 C[i][j] = 0;
			 for (int k = 0; k < NCOLUMNS; k++)
			 C[i][j] += A [i][k]*B[k][j];
		 }

}

void mmult_par(auto A[][NCOLUMNS], auto B[][NROWS], auto C[][NROWS], int thread_count){
	int i,j,k;
	#pragma omp parallel default(none) shared(A,B,C) private(i,j,k) num_threads(thread_count)
	{
 	#pragma omp for collapse(2) schedule(static)
	for (i = 0; i < NROWS; i++)
		 for (j = 0; j < NROWS; j++)
		 {
			 C[i][j] = 0;
			 for (k = 0; k < NCOLUMNS; k++)
			 C[i][j] += A [i][k]*B[k][j];
		 }

}
}
int main() 
{
    int i,k;
    struct timespec start, finish;
	double elapsed;
  	auto A = new int[NROWS][NCOLUMNS]();
  	auto B = new int[NCOLUMNS][NROWS]();
  	auto C = new int[NROWS][NROWS]();
  	 //NEW - sets the seed used by rand() to system clock

  	//initialize(A);
	 static int m = 1;
	for (i = 0; i < NROWS; i++)
	{
		srand(7*m+time(0));
		for (k = 0; k < NCOLUMNS; k++) 
		{
		A[i][k] = rand() % 100 + 1; 
		}
		m++;
	}
  	//initialize(B);
	for (i = 0; i < NCOLUMNS; i++)
	{
		srand(7*m+time(0));
		for (k = 0; k < NROWS; k++) 
		{
		B[i][k] = rand() % 100 + 1; 
		}
		m++;
	}


  //	print(A);


/*	cout<<"Initialized matrix A: "<<endl;
   for (i = 0; i < NROWS; ++ i){
  	cout<<endl;
     for (j = 0; j < NCOLUMNS; ++ j)
         cout<<A[i][j]<<"  ";
	}
//  	print(B);
	cout<<"Initialized matrix B: "<<endl;
   for (i = 0; i < NCOLUMNS; ++ i){
  	cout<<endl;
     for (j = 0; j < NROWS; ++ j)
         cout<<B[i][j]<<"  ";

     

}*/
clock_gettime(CLOCK_MONOTONIC, &start);
mmult_seq(A,B,C);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken seq: "<<elapsed<<" sec."<<endl;
clock_gettime(CLOCK_MONOTONIC, &start);
mmult_par(A,B,C,NUM_T);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken parallel with "<< NUM_T <<" threads: "<<elapsed<<" sec."<<endl;



/*
int cnt =0;
cout<<"Result matrix C: "<<endl;
   for (i = 0; i < NROWS; ++ i){
  	cout<<endl;
     for (int j = 0; j < NROWS; ++ j)
         {
//         	cout<<C[i][j]<<"  ";
         	cnt++;
         }
}
cout<<"Number of elements:"<< cnt;
*/
/*for (int i = 0; i < NROWS; ++i)
    delete [] A[i];*/
delete [] A;
/*for (int i = 0; i < NCOLUMNS; ++i)
    delete [] B[i];*/
delete [] B;
/*for (int i = 0; i < NROWS; ++i)
    delete [] C[i];*/
delete [] C;
}

   
   

