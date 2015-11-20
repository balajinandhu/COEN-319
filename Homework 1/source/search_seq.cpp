#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdlib>
#include<string>
#include<iterator>
#include<algorithm>
#include<time.h>
#include "boost/algorithm/searching/boyer_moore.hpp"
#include "boost/algorithm/string.hpp"

using namespace std;
using namespace boost;
using namespace boost::algorithm;



int main(int argc, char*argv[]){

	int l = 0;
	struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
	//read the text file from local drive
	ifstream in("haystack.txt");
	//store in string object
	string contents((istreambuf_iterator<char>(in)),
		istreambuf_iterator<char>());

	string contents_copy = contents.c_str();
	erase_all(contents_copy, " ");
	erase_all(contents_copy, "\n");
	erase_all(contents_copy, "\t");
	erase_all(contents_copy, "\r");
	erase_all(contents_copy, "\v");
	string pattern = argv[1];//"England";//"PoliticalEconomy"; //"couldn'tstandit";

    string::const_iterator beg  = contents.begin();
    string::const_iterator it_beg  = contents_copy.begin();
    string::const_iterator it_s  = contents_copy.begin();
    string::const_iterator ite_s = contents_copy.end();
    
    //vector<int> line_nos;
    int line_nos[1000] = {0};

    do{
    string::const_iterator it_res = boyer_moore_search<>(
             it_s, ite_s,				
             pattern.begin(),
             pattern.end());
	
    //to convert compressed text iterator to normal text iterator, 
    //calculate offset of the number of chars from beg to it_res 
    int offset = std::distance(it_beg, it_res);
    //move offset chars in contents ignoring whitespace - get iterator to normal text
    string::const_iterator il = contents.begin();
    int wcount=0;
    for(int j=0;j<=offset+wcount;++j) isspace(*il)? wcount++,*il++:*il++;
    if(it_res!=ite_s){
			
		int line_count = std::count(beg,il, '\n');
		line_nos[l] = line_count+1;
		++l;
	}
	advance(it_res, pattern.length());
	it_s = it_res;

	}while(it_s<ite_s);
	in.close();
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	if(l>0){
		//print number of occurences
		cout<<"Number of occurences of pattern in text: "<<l<<endl;
		//print line numbers 
		cout<<"The line numbers where the pattern occurs are: "<<endl;
		for(int i=0;i<l;i++){
			cout<<line_nos[i]<<endl;
		}
	}
	else{
		cout<<"NOT FOUND"<<endl;
	}
	cout<<"Time taken: "<<elapsed<<" sec."<<endl;
	return 1;
}