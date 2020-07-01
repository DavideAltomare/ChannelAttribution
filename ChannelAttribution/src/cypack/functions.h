#ifndef CALIB_H
#define CALIB_H

#include <Python.h>

#include <iostream>
#include <vector>
#include <set>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <sstream>
#include <list>
#include <armadillo>
#include <string>
#include <random>
#include <numeric>
#include <time.h> 
#include <thread>
#include <map>
#include <algorithm>
#include <list>

using namespace std;
using namespace arma;

pair < vector<string>,list< vector<double>> > heuristic_models_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, string sep);

//DEPRECATED
//pair < list< vector<string> >,list< vector<double> > > markov_model_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, vector<unsigned long int>& vn, unsigned long int order, unsigned long int nsim,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int seed);

list< vector<double> > choose_order_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt);

pair < list< vector<string> >,list< vector<double> > > markov_model_mp_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, vector<unsigned long int>& vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose);

pair < list< vector<string> >, vector<double> > transition_matrix_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int order, string sep, int flg_equal);

#endif