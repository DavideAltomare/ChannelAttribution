// ChannelAttribution: Markov model for online multi-channel attribution
// Copyright (C) 2015 -   Davide Altomare and David Loris <https://channelattribution.io>

// ChannelAttribution is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ChannelAttribution is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ChannelAttribution.  If not, see <http://www.gnu.org/licenses/>.


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

list< vector<double> > choose_order_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt);

pair < list< vector<string> >,list< vector<double> > > markov_model_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, vector<unsigned long int>& vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose);

pair < list< vector<string> >, vector<double> > transition_matrix_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int order, string sep, int flg_equal);

#endif
