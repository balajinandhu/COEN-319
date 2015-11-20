#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdlib>
#include<string>
#include<iterator>
#include<algorithm>
#include<pthread.h>
#include<time.h>
#include "boost/algorithm/searching/boyer_moore.hpp"
#include "boost/algorithm/string.hpp"

using namespace std;
using namespace boost;
using namespace boost::algorithm;
#define NUM_THREADS 8
#define NUM_LINES 12360

int line_nos[1000] = {0};
struct search_params
{
	
	int thread_id;
	string contents;
	string contents_copy[NUM_THREADS];
	string pattern;
	search_params(int tid, string c, string cc[NUM_THREADS], string pat) {
		thread_id = tid;
		contents = c;
		copy(cc, cc + NUM_THREADS, contents_copy);
		pattern = pat;
    }

};

pthread_mutex_t mutexsum;
pthread_t threads[NUM_THREADS];

void *searchString(void *thread_arg)
{
	static int l = 0;
	struct search_params *sarg;
	sarg = (struct search_params *) thread_arg;
    long tid = (long)sarg->thread_id;
    string::const_iterator beg  = sarg->contents.begin();
    string::const_iterator it_beg  = sarg->contents_copy[tid].begin();
    string::const_iterator it_s  = sarg->contents_copy[tid].begin();
    string::const_iterator ite_s = sarg->contents_copy[tid].end();
	    
   // Boyer moore search
    pthread_mutex_lock (&mutexsum);
	do{
	    string::const_iterator it_res = boyer_moore_search<>(
	             it_s, ite_s,				
	             sarg->pattern.begin(),
	             sarg->pattern.end());
		
	    int line_count = std::count(it_beg,it_res, '\n');
		if(it_res!=ite_s){
			
			line_nos[l] = tid*(NUM_LINES/NUM_THREADS)+line_count+1;
			++l;
			
		}
		advance(it_res, sarg->pattern.length());
		it_s = it_res;

	}while(it_s<ite_s);
	delete sarg;
   pthread_mutex_unlock (&mutexsum);
   pthread_exit((void *) thread_arg);
}

string findChunk(string str, int lines){
	static int i=0;
	string res_chunk;
	stringstream ss;
  	ss.str(str);
  	string temp;
  	int cnt = 0;
  	if (str != "")
  	{
  		for(int j=0;j<i*lines;j++){
  			std::getline(ss,temp,'\n');
  		}
    	while(std::getline(ss,temp,'\n')&&cnt<lines){
      		res_chunk.append(temp);
      		res_chunk.append("\n");
      		cnt++;
    }
  }
  i++;
  return res_chunk;
}



int main(int argc, char*argv[]){

	//read the text file from local drive
	struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
	ifstream in("haystack.txt");

	//store in string object
	string contents((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());

	string chunks[NUM_THREADS];
	string contents_copy = contents.c_str();
	erase_all(contents_copy, " ");
	erase_all(contents_copy, "\t");
	erase_all(contents_copy, "\r");
	erase_all(contents_copy, "\v");
	string pattern = argv[1];//"England";//"PoliticalEconomy"; //"couldn'tstandit";
	int rc;
	long t;
	
	//chunking
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		chunks[i] = findChunk(contents_copy, NUM_LINES/NUM_THREADS);
	}

	pthread_attr_t attr;
	void *status;

	pthread_mutex_init(&mutexsum, NULL);
	pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(t=0; t<NUM_THREADS; t++){
		search_params *sp = new search_params(
		t, 
		contents, 
  		chunks, 
  		pattern
  	);
      rc = pthread_create(&threads[t], &attr, searchString, (void *) sp);

      if (rc){
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

    	
   pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
      }
	in.close();
	pthread_mutex_destroy(&mutexsum);
	int count = 0;
	for(int i=0;i<1000&&line_nos[i]>0;i++) count++;
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	if(count>0){
		//print number of occurences
		cout<<"Number of occurences of pattern in text: "<<count<<endl;
		//print line numbers 
		cout<<"The line numbers where the pattern occurs are: "<<endl;
		for(int i=0;i<count;i++){
			cout<<line_nos[i]<<endl;
		}
	}
	else{
		cout<<"NOT FOUND"<<endl;
	}
	cout<<"Time taken: "<<elapsed<<" sec."<<endl;
	pthread_exit(NULL);
	return 1;
	
}