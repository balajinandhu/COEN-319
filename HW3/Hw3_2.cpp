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

void insertion_sort_seq(int *a, int n){
	int i,j,t;
	for(i=0;i<n;i++){
		t=a[i];
		for(j=i; j>0&&t<a[j-1];j--){
			a[j]=a[j-1];
		}
		a[j] = t;
	}
}
void insertionsort_helper(int *a, int n, int step) {
    for (int j=step; j<n; j+=step) {
        int key = a[j];
        int i = j - step;
        while (i >= 0 && a[i] > key) {
            a[i+step] = a[i];
            i-=step;
        }
        a[i+step] = key;
    }
}


void insertion_sort_par(int *a, int n){
  int i, j;

    for(j = n/2; j > 0; j /= 2)
    {
            #pragma omp parallel for default(none) shared(a,j,n) private (i) 
            for(i = 0; i < j; i++)
                insertionsort_helper(&(a[i]), n-i, j);
    }
}



int main(){
	struct timespec start, finish;
	double elapsed;
	int *Data = new int [NROWS];
	int *Data2 = new int [NROWS];

	static int m = 1;
	for (int i = 0; i < NROWS; i++)
	{
		srand(7*m+time(0));
		Data[i] = rand() % 100 + 1; 
		m++;
	}
/*     for (int j = 0; j < NROWS; ++ j)
         {
         	cout<<Data[j]<<"  ";
         }*/
copy(Data, Data+NROWS, Data2);
        

clock_gettime(CLOCK_MONOTONIC, &start);
insertion_sort_seq(Data,NROWS);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken seq: "<<elapsed<<" sec."<<endl;
/*cout<<"Result matrix Data: "<<endl;
     for (int j = 0; j < NROWS; ++ j)
         {
         	cout<<Data[j]<<"  ";
         }*/
clock_gettime(CLOCK_MONOTONIC, &start);
insertion_sort_par(Data2,NROWS);
clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
cout<<"Time taken parallel "<<elapsed<<" sec."<<endl;

//compare the two arrays 
     for (int j = 0; j < NROWS; ++ j)
         {
		if(Data[j]!=Data2[j])
        	cout<<"No match ";
         }
delete [] Data2;
delete [] Data;
return 1;

}
