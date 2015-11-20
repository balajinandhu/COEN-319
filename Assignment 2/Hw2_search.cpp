#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdlib>
#include<string>
#include<iterator>
#include<algorithm>
#include<pthread.h>
#include<time.h>
#include<list>
#include "boost/algorithm/searching/boyer_moore.hpp"
#include "boost/algorithm/string.hpp"

using namespace std;
using namespace boost;
using namespace boost::algorithm;
#define NUM_THREADS 8
#define NUM_LINES 12360

//template <typename T> 
class bqueue
{ 
    vector<string> m_queue;
    pthread_mutex_t m_mutex;
    pthread_cond_t  m_condv;

public:
  bqueue() {
      pthread_mutex_init(&m_mutex, NULL);
      pthread_cond_init(&m_condv, NULL);
  }
  ~bqueue() {
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_condv);
}

void add(string item) {
    pthread_mutex_lock(&m_mutex);
    m_queue.push_back(item);
    pthread_cond_signal(&m_condv);
    pthread_mutex_unlock(&m_mutex);
}
string remove() {
    pthread_mutex_lock(&m_mutex);
    while (m_queue.size() == 0) {
        pthread_cond_wait(&m_condv, &m_mutex);
    }
    string item = m_queue.front();
    m_queue.erase(m_queue.begin());
    pthread_mutex_unlock(&m_mutex);
    return item;
}
int size() {
        pthread_mutex_lock(&m_mutex);
        int size = m_queue.size();
        pthread_mutex_unlock(&m_mutex);
        return size;
    }
};

int line_nos[1000] = {0};
bqueue *block_queue = new bqueue();
struct search_params
{
	
	int thread_id;
	
	string pattern;
	search_params(int tid, string pat) {
		thread_id = tid;
		pattern = pat;
    }

};
struct producer_params
{
	ifstream in;
	producer_params(){
		ifstream in("haystack.txt");
	}
};

pthread_mutex_t mutexsum;
pthread_t threads[3];
pthread_t producer;

void *searchString(void *thread_arg)
{
	static int l = 0;
	struct search_params *sarg;
	sarg = (struct search_params *) thread_arg;
    long tid = (long)sarg->thread_id;
    string contents = block_queue->remove();
    string::const_iterator beg  = contents.begin();
    string contents_copy = contents.c_str();
	erase_all(contents_copy, " ");
	erase_all(contents_copy, "\t");
	erase_all(contents_copy, "\r");
	erase_all(contents_copy, "\v");
    string::const_iterator it_beg  = contents_copy.begin();
    string::const_iterator it_s  = contents_copy.begin();
    string::const_iterator ite_s = contents_copy.end();
	    
   // Boyer moore search
    pthread_mutex_lock (&mutexsum);
	do{
	    string::const_iterator it_res = boyer_moore_search<>(
	             it_s, ite_s,				
	             sarg->pattern.begin(),
	             sarg->pattern.end());
		
	    int line_count = std::count(it_beg,it_res, '\n');
		if(it_res!=ite_s){
			
			line_nos[l] = tid*1000+line_count+1;
			++l;
			
		}
		advance(it_res, sarg->pattern.length());
		it_s = it_res;

	}while(it_s<ite_s);
	delete sarg;
   pthread_mutex_unlock (&mutexsum);
   pthread_exit((void *) thread_arg);
}

void *findChunk(void *thread_arg){
	static int i=0;
	struct producer_params *sarg;
	sarg = (struct producer_params *) thread_arg;

	string res_chunk;

  	string temp;
  	int cnt = 0;
  
	for(int j=0;j<i*1000;j++){
			std::getline(sarg->in,temp,'\n');
	}
	while(std::getline(sarg->in,temp,'\n')&&cnt<1000){
  		res_chunk.append(temp);
  		res_chunk.append("\n");
  		cnt++;
	}

  i++;
  block_queue->add(res_chunk);	
  pthread_exit(NULL);
}



int main(int argc, char*argv[]){

	
	//read the text file from local drive
	struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
	


	//store in string object
	
	string pattern = argv[1];//"England";//"PoliticalEconomy"; //"couldn'tstandit";
	int rc;
	long t;
	
	
	producer_params *sp = new producer_params();


	//chunking producer thread
	 rc = pthread_create(&producer, NULL, findChunk, (void *) sp);

      if (rc){
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
	
	pthread_attr_t attr;
	void *status;

	pthread_mutex_init(&mutexsum, NULL);
	pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(t=0; t<3; t++){
		search_params *sp = new search_params(
		t, 
  		pattern
  	);
      rc = pthread_create(&threads[t], &attr, searchString, (void *) sp);

      if (rc){
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

    	
   pthread_attr_destroy(&attr);
    for(t=0; t<3; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
      }
	
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
	delete block_queue;
	return 1;
	
}