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
#define NROWS 10000
#define NCOLUMNS 1000

void loop_seq(int **Data, int n, int M){
	for(int i=1;i<n;i++){
		for(int j=1; j<M; j++){
			Data[i][j] = Data[i-1][j] + Data[i][j-1];
		}
	}
}

void loop_par(int **Data, int n, int M){
	int minor, major, diag, diagLength;
	if(n>M){
		minor = M;
		major = n;
	}
	else{
		minor = n;
		major = M;
	}
	for(diag = 1; diag <= n+M-3;diag++){
		diagLength = diag;
		if(diag+1>=minor)
			diagLength = minor-1;
		if(diag+1 >= major)
			diagLength = (minor - 1) - (diag - major) - 1;
		int i,j,k;
		#pragma omp parallel for shared(Data,n) private(k,i,j) schedule(static)
			for(k = 0; k < diagLength; k++){
				i = diag - k;
				j = k + 1;
				if(diag > n - 1){
					i = n - 1 - k;
					j = diag - (n - 1) + k + 1;
				}
	
				Data[i][j] = Data[i-1][j] + Data[i][j-1];

			}
	}
}

int main(){
	struct timespec start, finish;
	double elapsed;
	int **Data = new int *[NROWS];
        int **Data2 = new int *[NROWS];

	for(int i=0;i<NROWS;i++){
		Data[i] = new int[NCOLUMNS];
		Data2[i] = new int[NCOLUMNS];
	}


	static int m = 1;
	for (int i = 0; i < NROWS; i++)
	{
		srand(7*m+time(0));
		for (int k = 0; k < NCOLUMNS; k++) 
		{
		Data[i][k] = rand() % 100 + 1; 
		Data2[i][k] = Data[i][k];
		}
		m++;
	}
	Data[0][0] = 5;
	Data[0][1] = 5;
	Data[1][0] = 5;
	Data2[0][0] = 5;
	Data2[0][1] = 5;
	Data2[1][0] = 5;

clock_gettime(CLOCK_MONOTONIC, &start);
loop_seq(Data,NROWS,NCOLUMNS);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken seq: "<<elapsed<<" sec."<<endl;
clock_gettime(CLOCK_MONOTONIC, &start);
loop_par(Data2,NROWS,NCOLUMNS);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken parallel "<<elapsed<<" sec."<<endl;

for(int i=0;i<NROWS;i++)
	for(int j=0;j<NCOLUMNS;j++)
		if(Data[i][j]!=Data2[i][j])
			cout<<"No match";

/*
cout<<"Result matrix Data: "<<endl;
   for (int i = 0; i < NROWS; ++ i){
  	cout<<endl;
     for (int j = 0; j < NROWS; ++ j)
         {
       //  	cout<<Data[i][j]<<"  ";
         }
}*/
for(int i=0;i<NROWS;i++)
	delete [] Data[i];
delete [] Data;
for(int i=0;i<NROWS;i++)
	delete [] Data2[i];
delete [] Data2;

return 1;

}
