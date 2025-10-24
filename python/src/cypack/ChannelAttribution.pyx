# ChannelAttribution: Markov model for online multi-channel attribution
# Copyright (C) 2015 - 2025  Davide Altomare and David Loris <https://channelattribution.io>

# ChannelAttribution is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ChannelAttribution is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ChannelAttribution.  If not, see <http://www.gnu.org/licenses/>.


from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.pair cimport pair
from libc.time cimport time_t, tm, mktime

import pandas as pd
import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt 
import importlib

__version="2.2.2"
print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")
print("Version: " + str(__version))

cdef extern from "functions.h":
    pair[vector[string], list[vector[double]]] heuristic_models_cpp(vector[string]&, vector[unsigned long int]&, vector[double]&, string sep);
            
    list[vector[double]] choose_order_cpp(vector[string]& vy, vector[unsigned long int]& vc, vector[unsigned long int]& vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt);
    
    pair[list[vector[string]], list[vector[double]]] markov_model_cpp(vector[string]& vy, vector[unsigned long int]& vc, vector[double]& vv, vector[unsigned long int]& vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose);
    
    pair[list[vector[string]], vector[double]] transition_matrix_cpp(vector[string]& vy, vector[unsigned long int]& vc, vector[unsigned long int]& vn, unsigned long int order, string sep, int flg_equal)

def __heuristic_models_1(vector[string] vy, vector[unsigned long int] vc, vector[double] vv, string sep):
    return(heuristic_models_cpp(vy,vc,vv,sep))
    
def __choose_order_1(vector[string] vy, vector[unsigned long int] vc, vector[unsigned long int] vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt):
    return(choose_order_cpp(vy,vc,vn,max_order,sep,ncore,roc_npt))
    
def __markov_model_1(vector[string] vy, vector[unsigned long int] vc, vector[double] vv, vector[unsigned long int] vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose):
    return(markov_model_cpp(vy,vc,vv,vn,order,nsim_start,max_step,out_more,sep,ncore,nfold,seed,conv_par,rate_step_sim,verbose))
    
def __transition_matrix_1(vector[string] vy, vector[unsigned long int] vc, vector[unsigned long int] vn, unsigned long int order, string sep, int flg_equal):
    return(transition_matrix_cpp(vy,vc,vn,order,sep,flg_equal))

    
#https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b    
    
#start py

"""

**Markov Model for Online Multi-Channel Attribution**
Advertisers use a variety of online marketing channels to reach consumers and they want to know the degree each channel contributes to their marketing success. This is called online multichannel attribution problem. In many cases, advertisers approach this problem through some simple heuristics methods that do not take into account any customer interactions and often tend to underestimate the importance of small channels in marketing contribution. This package provides a function that approaches the attribution problem in a probabilistic way. It uses a k-order Markov representation to identify structural correlations in the customer journey data. This would allow advertisers to give a more reliable assessment of the marketing contribution of each channel. The approach basically follows the one presented in Eva Anderl, Ingo Becker, Florian v. Wangenheim,
Jan H. Schumann (2014). Differently from them, we solved the estimation process using stochastic simulations. In this way it is also possible to take into account conversion values and their variability in the computation of the channel importance. The package also contains a function that estimates three heuristic models (first-touch, last-touch and linear-touch approach) for the same problem.


"""
    
def heuristic_models(Data,var_path,var_conv,var_value=None, sep=">", flg_adv=True):

    """
            
    Estimate three heuristic models (first-touch, last-touch and linear) from customer journey data.
    
    Parameters
    ----------
    Data : DataFrame
        customer journeys.
    var_path: string
        column of Data containing paths.
    var_conv : string
        column of Data containing total conversions for each path.
    var_value : string, optional, default None
        column of Data containing revenue for each path.
    sep : string, default ">"
        separator between the channels.
    flg_adv : bool, default True
        if True, ChannelAttribution Pro banner is printed.
    
    Returns
    -------
    DataFrame        
        (column) channel_name : channel names
        (column) first_touch_conversions : conversions attributed to each channel using first touch attribution.
        (column) first_touch_value : revenues attributed to each channel using first touch attribution.
        (column) last_touch_conversions : conversions attributed to each channel using last touch attribution.
        (column) last_touch_value : revenues attributed to each channel using last touch attribution.
        (column) linear_touch_conversions : conversions attributed to each channel using linear attribution.
        (column) linear_touch_value : revenues attributed to each channel using linear attribution.
    
    Examples
    --------
    
    Load Data

    >>> import pandas as pd    
    >>> from ChannelAttribution import *
    >>> Data = pd.read_csv('https://channelattribution.io/csv/Data.csv',sep=";")
        
    Estimate heuristic models on total conversions
    
    >>> heuristic_models(Data,"path","total_conversions")
    
    Estimate heuristic models on total conversions and total revenues
    
    >>> heuristic_models(Data,"path","total_conversions",\\
    >>> var_value="total_conversion_value")
     
    """

    if ("DataFrame" not in str(type(Data))):
        raise NameError("Data must be a DataFrame")
    
    if type(var_path)==str:
        if var_path not in Data.columns:
            raise NameError("var_path must be a column of Data")
    else:
        raise NameError("var_path must be a string")

    if (type(var_conv)==str):
        if (var_conv not in Data.columns):
            raise NameError("var_conv must be a column of Data")
        
    else:
        raise NameError("var_conv must be a string")
   
    if (var_value!=None):
        if (var_value not in Data.columns):
            raise NameError("var_value must be a column of Data")
            
    if (len(sep) > 1):
        raise NameError("sep must have length 1")
        
    if (var_value==None):
        vv = pd.Series(None,dtype='float64')
    else:
        vv=Data[var_value]


    res0=__heuristic_models_1(Data[var_path].str.encode('utf-8'),Data[var_conv],vv,sep.encode("utf-8"))
    
    if len(vv)==0:
    
        res=pd.DataFrame({'channel_name':pd.Series(res0[0]).str.decode('utf-8'),'first_touch':res0[1][0],'last_touch':res0[1][1],'linear_touch':res0[1][2]})
    
    else:
    
        res=pd.DataFrame({'channel_name':pd.Series(res0[0]).str.decode('utf-8'),'first_touch_conversions':res0[1][0], 'first_touch_value':res0[1][3], 'last_touch_conversions':res0[1][1], 'last_touch_value':res0[1][4], 'linear_touch_conversions':res0[1][2], 'linear_touch_value':res0[1][5]})
    
    if flg_adv==True:
        print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")

    return(res)

    
def choose_order(Data,var_path,var_conv,var_null,max_order=10,sep=">",ncore=1,roc_npt=100,plot=True, flg_adv=True):

    """
    
    Find the minimum Markov Model order that gives a good representation of customers’ behaviour for data considered. It requires paths that do not lead to conversion as input. Minimum order is found maximizing a penalized area under ROC curve.
    
    Parameters
    ----------
    Data : DataFrame
        customer journeys.
    var_path: string
        column of Data containing paths.
    var_conv : string
        column of Data containing total conversions for each path.
    var_null : string
        column of Data containing total paths that do not lead to conversion.
    max_order : int, default 10
        maximum Markov Model order to be considered.        
    sep : string, default ">"
        separator between the channels.    
    ncore : int, default 1
        number of threads to be used in computation.        
    roc_npt: int, default 100
        number of points to be used for the approximation of roc curve.    
    plot: bool, default True
        if True, a plot with penalized auc with respect to order will be displayed.
    flg_adv : bool, default True
        if True, ChannelAttribution Pro banner is printed.
    
    Returns
    -------
    list        
        roc : list DataFrame one for each order considered
            (column) tpr: true positive rate
            (column) fpr: false positive rate
        auc : DataFrame with the following columns
            (column) order: markov model order  
            (column) auc: area under the curve
            (column) pauc: penalized auc
        suggested order : int
            estimated best order 
            
    Examples
    --------
    Estimate best makov model order for your data
    
    Load Data
        
    >>> import pandas as pd    
    >>> from ChannelAttribution import *
    >>> Data = pd.read_csv('https://channelattribution.io/csv/Data.csv',sep=";")
    
    >>> choose_order(Data, var_path="path", var_conv="total_conversions", var_null="total_null")


    """
    
    if ("DataFrame" not in str(type(Data))):
        raise NameError("Data must be a DataFrame")
    
    if type(var_path)==str:
        if var_path not in Data.columns:
            raise NameError("var_path must be a column of Data")
    else:
        raise NameError("var_path must be a string")

    if (type(var_conv)==str):
        if (var_conv not in Data.columns):
            raise NameError("var_conv must be a column of Data")
        
    else:
        raise NameError("var_conv must be a string")
   
        
    if (var_null!=None):
        if (var_null not in Data.columns):
            raise NameError("var_null must be a column of Data")
      
    if (max_order < 1):
        raise NameError("max_order must be >= 1")
    
    if (ncore!=None):
        if (ncore < 1):
            raise NameError("ncore must be >= 1")
            
    if (roc_npt!=None):
        if (roc_npt < 10):
            raise NameError("roc_npt must be >= 10")
        
    
    if (plot not in [0, 1]):
        raise NameError("plot must be False or True")
    
    res0=__choose_order_1(vy=Data[var_path].str.encode('utf-8'),vc=Data[var_conv],vn=Data[var_null], max_order=max_order, sep=sep.encode('utf-8'), ncore=ncore, roc_npt=roc_npt)    

    order=pd.Series(res0[-3])
    auc=pd.Series(res0[-2])
    pauc=pd.Series(res0[-1])
 
    max_order_0=order[order!=0].iloc[-1]
    
    best_order=res0[-1].index(max(res0[-1]))
   
    best_order=best_order+1

    if best_order==max_order_0:
        print("Suggested order not found. Try increasing max_order.")
    else:
        print("Suggested order: " + str(int(best_order)))

    auc=auc[order!=0]
    pauc=pauc[order!=0]
    order=order[order!=0]
        
    if plot=="True":
        plt.title("PENALIZED AUC")
        plt.xlabel("order")
        plt.ylabel("penalized auc")
        plt.plot(order, pauc) 
    
    auc=auc[order<=(best_order+1)]
    pauc=pauc[order<=(best_order+1)]
    order=order[order<=(best_order+1)]
    
    res_auc=pd.DataFrame({'order':order,'auc':auc,'pauc':pauc})
    
    res_roc=dict()
    for k in range(best_order+1):
        res_roc['order='+str(k+1)]=pd.DataFrame({'fpr':res0[2*k],'tpr':res0[2*k+1]})
    
    if flg_adv==True:
        print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")
    
    return(res_auc,res_roc,best_order)
            
    
        
def markov_model(Data,var_path,var_conv,var_value=None,var_null=None,order=1,nsim_start=1e5,max_step=None,out_more=False,sep=">",ncore=1, nfold=10, seed=0, conv_par=0.05,rate_step_sim=1.5,verbose=True, flg_adv=True):

    '''
    
    Estimate a k-order Markov model from customer journey data. Differently from markov_model, this function iterates estimation until a desidered convergence is reached and enables multiprocessing.
    
    Parameters
    ----------
    Data : DataFrame
        customer journeys.
    var_path: string
        column of Data containing paths.
    var_conv : string
        column of Data containing total conversions for each path.
    var_value : string, optional, default None
        column of Data containing revenue for each path.
    var_null : string
        column of Data containing total paths that do not lead to conversion.
    order : int, default 1
        Markov model order.        
    nsim_start : int, default 1e5
        minimum number of simulations to be used in computation.        
    max_step : int, default None
        maximum number of steps for a single simulated path. if NULL, it is the maximum number of steps found into Data.        
    out_more : bool, default False
        if True, transition probabilities between channels and removal effects will be returned.                
    sep : string, default ">"
        separator between the channels.    
    ncore : int, default 1
        number of threads to be used in computation.        
    nfold : int, default 10
        how many repetitions to be used to verify if convergence has been reached at each iteration.    
    seed : int, default 0
        random seed. Giving this parameter the same value over different runs guarantees that results will not vary.    
    conv_par : double, default 0.05
        convergence parameter for the algorithm. The estimation process ends when the percentage of variation of the results over different repetions is less than convergence parameter.    
    rate_step_sim : double, default 0
        number of simulations used at each iteration is equal to the number of simulations used at previous iteration multiplied by rate_step_sim.    
    verbose : bool, default True
        if True, additional information about process convergence will be shown.    
    flg_adv : bool, default True
        if True, ChannelAttribution Pro banner is printed.
            
    Returns
    -------
    list of DataFrames
        result: Dataframe
            (column) channel_name : channel names
            (column) total_conversions : conversions attributed to each channel
            (column) total_conversion_value : revenues attributed to each channel
        transition_matrix : DataFrame
            (column) channel_from: channel from
            (column) channel_to : channel to
            (column) transition_probability : transition probability from channel_from to channel_to
        removal_effects:
            (column) channel_name : channel names 
            (column) removal_effects_conversion : removal effects for each channel calculated using total conversions
            (column) removal_effects_conversion_value : removal effects for each channel calculated using revenues
                
                        
    Examples
    --------
    
    Load Data
    
    >>> import pandas as pd    
    >>> from ChannelAttribution import *
    >>> Data = pd.read_csv('https://channelattribution.io/csv/Data.csv',sep=";")
    
    Estimate a Makov model using total conversions 
    
    >>> markov_model(Data, "path", "total_conversions")

    Estimate a Makov model using total conversions and revenues 
    
    >>> markov_model(Data, "path", "total_conversions", var_value="total_conversion_value")
    
    Estimate a Makov model using total conversions, revenues and paths that do not lead to conversions 

    >>> markov_model(Data, "path", "total_conversions", var_value="total_conversion_value", var_null="total_null")
    
    Estimate a Makov model returning transition matrix and removal effects 
    
    >>> markov_model(Data, "path", "total_conversions", var_value="total_conversion_value", var_null="total_null", out_more=True)

    Estimate a Markov model using 4 threads
    
    >>> markov_model(Data, "path", "total_conversions", var_value="total_conversion_value", ncore=4)
        
    '''

    if ("DataFrame" not in str(type(Data))):
        raise NameError("Data must be a DataFrame")
    
    if type(var_path)==str:
        if var_path not in Data.columns:
            raise NameError("var_path must be a column of Data")
    else:
        raise NameError("var_path must be a string")

    if (type(var_conv)==str):
        if (var_conv not in Data.columns):
            raise NameError("var_conv must be a column of Data")
        
    else:
        raise NameError("var_conv must be a string")
   
    if (var_value!=None):
        if (var_value not in Data.columns):
            raise NameError("var_value must be a column of Data")
        
    if (var_null!=None):
        if (var_null not in Data.columns):
            raise NameError("var_null must be a column of Data")
      
    if (order < 1):
        raise NameError("order must be >= 1")
    
    if (nsim_start!=None):
        if (nsim_start < 1):
            raise NameError("nsim must be >= 1")
        
    if (max_step!=None):
        if (max_step < 1):
            raise NameError("max_step must be >= 1")
        
    
    if (out_more not in [0, 1]):
        raise NameError("out_more must be False or True")
    
        
    if (len(sep) > 1):
        raise NameError("sep must have length 1")

    
    if (ncore < 1):
        raise NameError("ncore must be >= 1")
    
    if (nfold < 1):
        raise NameError("nfold must be >= 1")
    
    if (seed!=None):
        if (seed < 0):
            raise NameError("seed must be >= 0")

    if ((conv_par < 0) | (conv_par > 1)):
        raise NameError("conv_par must be into [0,1]")


    if (rate_step_sim < 0):
        raise NameError("rate_step_sim must be > 0")
        
    if (verbose not in [0, 1]): 
        raise NameError("verbose must be False or True")
        
        
    if sum(Data[var_conv]>0)==0:
        raise NameError("Data must have at least one converting path.")

    if (var_null==None):
        Data=Data[Data[var_conv]>0]    
    
    if (var_value==None):
        vv = pd.Series(None,dtype='float64')
    else:
        vv=Data[var_value]
    

    if (var_null==None):
        vn = pd.Series(None,dtype='float64')
    else:
        vn=Data[var_null]
        
    if (max_step==None):
        max_step = 0
    
    res0=__markov_model_1(vy=Data[var_path].str.encode('utf-8'),vc=Data[var_conv],vv=vv,vn=vn,order=order,nsim_start=nsim_start,max_step=max_step,out_more=out_more,sep=sep.encode("utf-8"),ncore=ncore, nfold=nfold, seed=seed, conv_par=conv_par, rate_step_sim=rate_step_sim, verbose=int(verbose))
        
    if (out_more==0) and (len(vv)==0):
    
        res=pd.DataFrame({'channel_name':pd.Series(res0[0][0]).str.decode('utf-8'),'total_conversions':res0[1][0]})
    
    elif (out_more==0) and (len(vv)>0):

        res=pd.DataFrame({'channel_name':pd.Series(res0[0][0]).str.decode('utf-8'),'total_conversions':res0[1][0],'total_conversion_value':res0[1][1]})

    elif (out_more==1) and (len(vv)==0):
        
        res=dict()
        
        res['result']=pd.DataFrame({'channel_name':pd.Series(res0[0][2]).str.decode('utf-8'),'total_conversions':res0[1][0]})
        res['transition_matrix']=pd.DataFrame({'channel_from':pd.Series(res0[0][0]).str.decode('utf-8'),'channel_to':pd.Series(res0[0][1]).str.decode('utf-8'),'transition_probability':res0[1][2]})
    
        res['removal_effects']=pd.DataFrame({'channel_name':pd.Series(res0[0][2]).str.decode('utf-8'),'removal_effect':res0[1][1]})
    
    else:
    
        res=dict()
        
        res['result']=pd.DataFrame({'channel_name':pd.Series(res0[0][2]).str.decode('utf-8'),'total_conversions':res0[1][0],'total_conversion_value':res0[1][3]})
        res['transition_matrix']=pd.DataFrame({'channel_from':pd.Series(res0[0][0]).str.decode('utf-8'),'channel_to':pd.Series(res0[0][1]).str.decode('utf-8'),'transition_probability':res0[1][2]})
    
        res['removal_effects']=pd.DataFrame({'channel_name':pd.Series(res0[0][2]).str.decode('utf-8'),'removal_effects_conversion':res0[1][1],'removal_effects_conversion_value':res0[1][4]})        
    
    if flg_adv==True:
        print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")
    
    return(res)
    

def transition_matrix(Data,var_path,var_conv,var_null,order=1,sep=">",flg_equal=True, flg_adv=True):

    '''

    Estimate a k-order transition matrix from customer journey data.
    
    Parameters
    ----------
    Data : DataFrame
        customer journeys.
    var_path: string
        column of Data containing paths.
    var_conv : string
        column of Data containing total conversions for each path.
    var_null : string
        column of Data containing total paths that do not lead to conversion.
    order : int, default 1
        Markov model order.        
    sep : string, default ">"
        separator between the channels.    
    flg_equal: bool, default True
        if True, transitions from a channel to itself will be considered.    
    flg_adv : bool, default True
        if True, ChannelAttribution Pro banner is printed.
                    
    Returns
    -------
    list of DataFrames
        channels: Dataframe
            (column) id_channel : channel ids
            (column) channel_name : channel names
        transition_matrix : DataFrame
            (column) channel_from: id channel from
            (column) channel_to : id channel to
            (column) transition_probability : transition probability from channel_from to channel_to
                    
    Examples
    --------
    
    Load Data
    
    >>> import pandas as pd    
    >>> from ChannelAttribution import *
    >>> Data = pd.read_csv('https://channelattribution.io/csv/Data.csv',sep=";")

    Estimate a second-order transition matrix using total conversions and paths that do not lead to conversion 
    
    >>> transition_matrix(Data, "path", "total_conversions", var_null="total_null", order=2)
                    
    '''
     
    if ("DataFrame" not in str(type(Data))):
        raise NameError("Data must be a DataFrame")
    
    if type(var_path)==str:
        if var_path not in Data.columns:
            raise NameError("var_path must be a column of Data")
    else:
        raise NameError("var_path must be a string")

    if (type(var_conv)==str):
        if (var_conv not in Data.columns):
            raise NameError("var_conv must be a column of Data")
    else:
        raise NameError("var_conv must be a string")

    if (type(var_null)==str):
        if (var_null not in Data.columns):
            raise NameError("var_null must be a column of Data")        
    else:
        raise NameError("var_conv must be a string")
   
      
    if (order < 1):
        raise NameError("order must be >= 1")
    
        
    if (len(sep) > 1):
        raise NameError("sep must have length 1")

            
    if (flg_equal not in [0, 1]): 
        raise NameError("flg_equal must be False or True")
                    
    res0=__transition_matrix_1(vy=Data[var_path].str.encode('utf-8'),vc=Data[var_conv],vn=Data[var_null],order=order,sep=sep.encode("utf-8"), flg_equal=int(flg_equal))
    
    res=dict()
    res['channels']=pd.DataFrame({'id_channel':range(1,len(res0[0][2])+1), 'channel_name':pd.Series(res0[0][2]).str.decode('utf-8')})
    res['transition_matrix']=pd.DataFrame({'channel_from':pd.Series(res0[0][0]).str.decode('utf-8'),'channel_to':pd.Series(res0[0][1]).str.decode('utf-8'),'transition_probability':res0[1]})
    
    if flg_adv==True:
        print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")

    return(res)
    
    
    
def auto_markov_model(Data, var_path, var_conv, var_null, var_value=None, max_order=10, roc_npt=100, plot=False, nsim_start=1e5, max_step=None, out_more=False, sep=">", ncore=1, nfold=10, seed=0, conv_par=0.05, rate_step_sim=1.5, verbose=True, flg_adv=True):

    '''
    
    Parameters
    ----------
    Data : DataFrame
        customer journeys.
    var_path: string
        column of Data containing paths.
    var_conv : string
        column of Data containing total conversions for each path.
    var_null : string
        column of Data containing total paths that do not lead to conversion.
    var_value : string, optional, default None
        column of Data containing revenue for each path
    max_order : int, default 10
        maximum Markov Model order to be considered.        
    roc_npt: int, default 100
        number of points to be used for the approximation of roc curve.    
    plot: bool, default True
        if True, a plot with penalized auc with respect to order will be displayed.
    nsim_start : int, default 1e5
        minimum number of simulations to be used in computation.        
    max_step : int, default None
        maximum number of steps for a single simulated path. if NULL, it is the maximum number of steps found into Data.        
    out_more : bool, default False
        if True, transition probabilities between channels and removal effects will be returned.                
    sep : string, default ">"
        separator between the channels.    
    ncore : int, default 1
        number of threads to be used in computation.        
    nfold : int, default 10
        how many repetitions to be used to verify if convergence has been reached at each iteration.    
    seed : int, default 0
        random seed. Giving this parameter the same value over different runs guarantees that results will not vary.    
    conv_par : double, default 0.05
        convergence parameter for the algorithm. The estimation process ends when the percentage of variation of the results over different repetions is less than convergence parameter.    
    rate_step_sim : double, default 0
        number of simulations used at each iteration is equal to the number of simulations used at previous iteration multiplied by rate_step_sim.    
    verbose : bool, default True
        if True, additional information about process convergence will be shown.    
    flg_adv : bool, default True
        if True, ChannelAttribution Pro banner is printed.
            
    Returns
    -------
    list of DataFrames
        result: Dataframe
            (column) channel_name : channel names
            (column) total_conversions : conversions attributed to each channel
            (column) total_conversion_value : revenues attributed to each channel
        transition_matrix : DataFrame
            (column) channel_from: channel from
            (column) channel_to : channel to
            (column) transition_probability : transition probability from channel_from to channel_to
        removal_effects:
            (column) channel_name : channel names 
            (column) removal_effects_conversion : removal effects for each channel calculated using total conversions
            (column) removal_effects_conversion_value : removal effects for each channel calculated using revenues
                
                        
    Examples
    --------
    
    Load Data
    
    >>> import pandas as pd    
    >>> from ChannelAttribution import *
    >>> Data = pd.read_csv('https://channelattribution.io/csv/Data.csv',sep=";")
    
    Estimate an automatic Makov model 
    
    >>> auto_markov_model(Data, "path", "total_conversions", "total_null")
        
    '''

    if ("DataFrame" not in str(type(Data))):
        raise NameError("Data must be a DataFrame")
    
    if type(var_path)==str:
        if var_path not in Data.columns:
            raise NameError("var_path must be a column of Data")
    else:
        raise NameError("var_path must be a string")

    if (type(var_conv)==str):
        if (var_conv not in Data.columns):
            raise NameError("var_conv must be a column of Data")
        
    else:
        raise NameError("var_conv must be a string")
   
   
    if (type(var_null)==str):
        if (var_null not in Data.columns):
            raise NameError("var_null must be a column of Data")
        
    else:
        raise NameError("var_null must be a string")
   
   
    if (var_value!=None):
        if (var_value not in Data.columns):
            raise NameError("var_value must be a column of Data")
              
    if (max_order < 1):
        raise NameError("max_order must be >= 1")
        
    if (roc_npt < 10):
        raise NameError("roc_npt must be >= 10")
        
    if (plot not in [0, 1]):
        raise NameError("plot must be False or True")
    
    if (nsim_start!=None):
        if (nsim_start < 1):
            raise NameError("nsim must be >= 1")
        
    if (max_step!=None):
        if (max_step < 1):
            raise NameError("max_step must be >= 1")
        
    
    if (out_more not in [0, 1]):
        raise NameError("out_more must be False or True")
    
        
    if (len(sep) > 1):
        raise NameError("sep must have length 1")

    
    if (ncore < 1):
        raise NameError("ncore must be >= 1")
    
    if (nfold < 1):
        raise NameError("nfold must be >= 1")
    
    if (seed!=None):
        if (seed < 0):
            raise NameError("seed must be >= 0")

    if ((conv_par < 0) | (conv_par > 1)):
        raise NameError("conv_par must be into [0,1]")


    if (rate_step_sim < 0):
        raise NameError("rate_step_sim must be > 0")
        
    if (verbose not in [0, 1]): 
        raise NameError("verbose must be False or True")
    
    if (var_value==None):
        vv = pd.Series(None,dtype='float64')
    else:
        vv=Data[var_value]
            
    [res_auc,res_roc,best_order] = choose_order(Data, var_path, var_conv, var_null, max_order = max_order, sep = sep, ncore = ncore, roc_npt = roc_npt, plot = plot, flg_adv=False)
    
    res = markov_model(Data, var_path, var_conv, var_value = var_value, var_null = var_null, order = best_order, nsim_start = nsim_start, max_step = max_step, out_more = out_more, sep = sep, ncore = ncore, nfold = nfold, seed = seed, conv_par = conv_par, rate_step_sim = rate_step_sim, verbose = verbose, flg_adv=False)
    
    if flg_adv==True:
        print("*** Looking to run more advanced attribution? Try ChannelAttribution Pro for free! Visit https://channelattribution.io/product")
    
    return(res)



def request_token_channelattributionpro(
    email: str,
    endpoint: str = "https://app.channelattribution.io/genpkg/generate_token.php",
    timeout: int = 10,
    verify_ssl: bool = True
) -> str:

    """
    Send an email address to ChannelAttributionPro's `generate_token.php` endpoint and
    return the raw response body emitted by the server.

    Parameters
    ----------
    email : str
        Target email address to which the token should be sent. Must be non-empty and
        syntactically valid; otherwise a ``ValueError`` is raised.
    endpoint : str, default "https://app.channelattribution.io/genpkg/generate_token.php"
        Full URL of the token-generation PHP endpoint. You can override this for testing.
    timeout : int, default 10
        Timeout in seconds applied to the HTTP request(s).
    verify_ssl : bool, default True
        Whether to verify the server's TLS certificate. Set to ``False`` only for
        controlled testing environments.

    Returns
    -------
    str
        The exact response body (trimmed) returned by the server. Typical values include:
        - ``"We’ve sent the token to your email address. ..."``
        - ``"Token already generated"``
        - ``"Provider not admitted"``
        - ``"mail not valid"``
        - ``"db query error"``, ``"db connection error"``, etc.

    Raises
    ------
    ValueError
        If ``email`` is empty.
    RuntimeError
        For network or SSL issues (connection errors, DNS failure, timeouts, TLS problems),
        with a message prefixed by ``"network_or_ssl_error:"`` or ``"request_error:"``.

    Notes
    -----
    - The function **prefers POST** and will **retry with GET** if the server rejects the method
      (e.g., HTTP 405/403 with a “method” hint in the body).
    - The function does **not** raise for non-2xx HTTP statuses; it returns the body as-is so the
      calling code can display the server’s message to the user.

    Examples
    --------
    Basic usage

    >>> request_token_channelattributionpro("alice@example.com")
    'We’ve sent the token to your email address. Please check your Spam or Junk folder if it’s not in your inbox. If you still can’t find it, write to info@channelattribution.io.'

    Handling network errors

    >>> try:
    ...     request_token_channelattributionpro("alice@example.com", timeout=3)
    ... except RuntimeError as e:
    ...     print(e)  # e.g., "network_or_ssl_error: HTTPSConnectionPool(...): Read timed out."
    """

    import requests
    from requests.exceptions import RequestException, Timeout, SSLError

    if not email:
        raise ValueError("email must be a non-empty string")

    try:
        # Prefer POST
        resp = requests.post(
            endpoint,
            data={"email": email},
            timeout=timeout,
            verify=verify_ssl,
            allow_redirects=True,
            headers={"User-Agent": "capro-token-client/1.0"}
        )

        # If server disallows POST (rare), retry with GET
        if resp.status_code in (405, 403) and "method" in (resp.text or "").lower():
            resp = requests.get(
                endpoint,
                params={"email": email},
                timeout=timeout,
                verify=verify_ssl,
                allow_redirects=True,
                headers={"User-Agent": "capro-token-client/1.0"}
            )

        # We return the body regardless of status, as requested.
        # If you prefer to fail on non-2xx, uncomment the two lines below.
        # if not resp.ok:
        #     raise RuntimeError(f"Server returned HTTP {resp.status_code}: {resp.text.strip()}")

        return (resp.text or "").strip()

    except (Timeout, SSLError) as e:
        raise RuntimeError(f"network_or_ssl_error: {e}") from e
    except RequestException as e:
        # Covers connection errors, invalid URLs, etc.
        raise RuntimeError(f"request_error: {e}") from e


def install_channelattributionpro(token: str | None = None):

    '''
    Install ChannelAttribution Pro (binary wheel) for the current environment.

    This installer detects your OS, architecture, and Python version, requests a
    prebuilt wheel (or triggers a build) from ChannelAttribution Pro’s build
    service, resolves the final package URL, and installs it via `pip`. If
    installation fails, it prints a compact system report you can send to support.

    Parameters
    ----------
    token : str, optional
        Access token for the build service. If omitted, the function will read
        the environment variable `CHPRO_TOKEN`. If neither is provided, the
        installer prints a message and returns. If the token is invalid or
        expired, the installer prints **"token non valid or expired"** and
        returns.

    Environment detection
    ---------------------
    OS:
        - manylinux (Linux), macos (macOS), windows (Windows)
    OS version mapping:
        - macOS:   "13" for amd64 (Intel), "15" for arm64 (Apple Silicon)
        - Windows: "11"
        - Linux:   "2014" (ManyLinux2014 baseline)
    Architecture:
        - amd64 (x86_64)
        - arm64 (aarch64)
    Python:
        - Major.minor version detected from the running interpreter (e.g., 3.11)

    Behavior
    --------
    1) Builds a request URL to the ChannelAttribution Pro builder:
       https://app.channelattribution.io/genpkg/genpkg.php
       with the detected parameters **and the `token`**.
    2) Performs an HTTPS GET. The service may:
       - Return HTTP 200 with JSON containing "pkg": a direct wheel URL or a
         directory containing wheels.
       - Return HTTP 409 with JSON ("exists"/"ok") pointing to an existing build.
       - Return HTTP 401 if the token is invalid/expired. In this case the
         installer prints **"token non valid or expired"** and returns.
    3) Resolves the final wheel URL (if a directory is returned, picks the latest file).
    4) Installs the wheel with `python -m pip install --prefer-binary`.
    5) On success, prints a short message suggesting to restart the session and import:
       `import ChannelAttributionPro`.
    6) On failure (network/build issues), prints a JSON-like system info block
       (OS, distro, Python, compiler) that you can email to info@channelattribution.io.

    Network & security
    ------------------
    - Uses HTTPS GET to the builder endpoint.
    - Sends only non-personal environment traits (OS/arch/Python) and your **token**
      as query parameters.
    - You can pass the token as a function argument or via the `CHPRO_TOKEN`
      environment variable to avoid hardcoding in code/notebooks.
    - Honors standard proxy settings if your Python/OS is configured accordingly.

    Requirements
    ------------
    - Internet connectivity to reach app.channelattribution.io.
    - `pip` available for the current interpreter (`python -m pip`).
    - Sufficient permissions to install packages in the environment (use a venv or
      run with appropriate privileges).
    - A valid access token.

    Notes
    -----
    - Depending on load, the remote build step may take several minutes.
    - On macOS, `gcc` typically maps to Clang; compiler info is reported accordingly.
    - The Linux baseline targets ManyLinux2014 for broad compatibility.

    Returns
    -------
    None
        The function performs installation as a side effect and writes progress
        to stdout. On error, it prints diagnostic information and returns without
        raising exceptions.

    Exceptions
    ----------
    None raised by this function.
        All error conditions are handled by printing a message and returning.
        (Network errors, invalid token, missing/invalid response, and pip failures
        are reported via stdout.)

    Examples
    --------
    Basic usage with explicit token

    >>> from ChannelAttributionPro import install_channelattributionpro
    >>> install_channelattributionpro(token="YOUR_TOKEN_HERE")
    Building the package. Estimated time: 5-30 minutes. Please wait...
    ...
    Package installed. Restart the session and try to import it with: import ChannelAttributionPro

    Using environment variable

    # Linux/macOS:
    # export CHPRO_TOKEN=YOUR_TOKEN_HERE
    # Windows (new shells):
    # setx CHPRO_TOKEN YOUR_TOKEN_HERE
    >>> install_channelattributionpro()
    ...

    Invalid/expired token

    >>> install_channelattributionpro(token="bad_or_expired")
    token non valid or expired

    After installation

    >>> import ChannelAttributionPro
    >>> ChannelAttributionPro.__version__
    'x.y.z'
    '''

    import requests
    from requests.exceptions import RequestException, Timeout, SSLError
    
    def notify_package_request(
        token: str,
        action: str,
        endpoint: str = "https://app.channelattribution.io/genpkg/build_check_email.php",
        timeout: int = 10,
        verify_ssl: bool = True,
    ) -> str:
    
        if not token:
            return "missing_token_param"
    
        try:
            resp = requests.get(
                endpoint,
                params={"token": token, "action": action},
                timeout=timeout,
                verify=verify_ssl,
                allow_redirects=True,
                headers={"User-Agent": "capro-build-check/1.0"},
            )
            return (resp.text or "").strip()
        except (Timeout, SSLError) as e:
            return f"network_or_ssl_error: {e}"
        except RequestException as e:
            return f"request_error: {e}"
    
    resp=notify_package_request(token,"START")

    import os
    import sys
    import platform
    import json
    import subprocess
    from html.parser import HTMLParser
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    from urllib.parse import urlencode, urljoin

    # -------- Token handling (print-only) --------
    if token is None:
        token = os.environ.get("CHPRO_TOKEN")
    if not token:
        print("Missing token. Pass token=... or set CHPRO_TOKEN in the environment.")
        return

    # -------- Detect OS / arch / python --------
    if sys.platform.startswith("linux"):
        os_name = "manylinux"
    elif sys.platform == "darwin":
        os_name = "macos"
    elif sys.platform in ("win32", "cygwin", "msys"):
        os_name = "windows"
    else:
        os_name = "manylinux"

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = "amd64"

    lang = "python"
    lang_vers = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Fixed mapping for os_vers
    if os_name == "macos":
        os_vers = "13" if arch == "amd64" else "15"
    elif os_name == "windows":
        os_vers = "11"
    else:
        os_vers = "2014"

    params = {
        "os": os_name,
        "os_vers": os_vers,
        "arch": arch,
        "lang": lang,
        "lang_vers": lang_vers,
        "replace": "0",
        "uctr": "0",
        "token": token,
    }

    base_url = "https://app.channelattribution.io/genpkg/genpkg.php"
    gen_url = f"{base_url}?{urlencode(params)}"

    UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    HEADERS = {
        "User-Agent": UA,
        "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.7",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    def http_get(u, timeout=300):
        try:
            req = Request(u, headers=HEADERS)
            with urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), resp.headers
        except HTTPError as e:
            # Return the code and body so caller can inspect without raising
            try:
                body = e.read()
            except Exception:
                body = b""
            return e.code, body, getattr(e, "headers", {})
        except URLError as e:
            # Network failure — status 0
            return 0, str(e).encode("utf-8", errors="replace"), {}

    class LinkCollector(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag.lower() == "a":
                href = dict(attrs).get("href")
                if href:
                    self.links.append(href)

    def list_dir_files(dir_url):
        status, body, _headers = http_get(dir_url)
        if status != 200:
            print(f"Listing {dir_url} failed with HTTP {status}")
            return None
        html = body.decode("utf-8", errors="replace")
        p = LinkCollector()
        p.feed(html)
        return [h for h in p.links if h and h not in ("/", "../") and not h.endswith("/")]

    def resolve_pkg_url(pkg_value):
        """
        Accepts either a wheel URL or a directory URL and returns a wheel URL.
        Print-only failure; returns None if cannot resolve.
        """
        if not isinstance(pkg_value, str) or not pkg_value:
            print("Invalid 'pkg' value in response.")
            return None

        if pkg_value.lower().endswith(".whl"):
            return pkg_value

        # treat as directory
        pkg_dir = pkg_value.rstrip("/") + "/"
        files = list_dir_files(pkg_dir)
        if files is None:
            return None
        wheels = [f for f in files if f.endswith(".whl")]
        chosen = (sorted(wheels) or sorted(files) or [None])[-1]
        if not chosen:
            print(f"No files found at {pkg_dir}")
            return None
        return urljoin(pkg_dir, chosen)

    def pip_install(url, extra_args=None):
        if not url:
            print("No package URL to install.")
            return False
        cmd = [sys.executable, "-m", "pip", "install",
               "--no-cache-dir", "--disable-pip-version-check", "--prefer-binary", url]
        if extra_args:
            cmd.extend(extra_args)
        print("Installing with:", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"pip failed with exit code {proc.returncode}")
            return False
        return True

    # -------- main flow --------
    print("Building the package. Estimated time: 0-30 minutes. Please wait...")
    status, body, headers = http_get(gen_url)
    text = body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)

    # 401 → token invalid (exact message requested)
    if status == 401:
        print("Token non valid or expired.")
        return

    pkg_file_url = None
    flg_success = 1

    # Try to parse JSON if possible
    data = None
    try:
        data = json.loads(text)
    except Exception:
        data = None

    # If JSON indicates token failure despite 200 (defensive)
    if isinstance(data, dict):
        err = (data.get("error") or "").lower()
        stat = (data.get("status") or "").lower()
        if "invalid token" in err or (stat in ("fail", "error") and "token" in err):
            print("token non valid or expired")
            return

    # Happy paths
    if status == 200 and isinstance(data, dict) and "pkg" in data:
        pkg_file_url = resolve_pkg_url(data["pkg"])
        if not pkg_file_url:
            flg_success = 0
    elif status == 409 and isinstance(data, dict) and data.get("status") in ("exists", "ok") and "pkg" in data:
        pkg_file_url = resolve_pkg_url(data["pkg"])
        if not pkg_file_url:
            flg_success = 0
    else:
        flg_success = 0
        # print a concise server hint
        if status == 0:
            print(f"Network error while contacting builder: {text[:500]}")
        else:
            print(f"Unexpected response from builder (HTTP {status}). Body: {text[:500]}")

    if flg_success and pkg_file_url:
        ok = pip_install(pkg_file_url)
        if ok:
            print("Package installed. Restart the session and try to import it with: import ChannelAttributionPro")
            resp=notify_package_request(token,"END")
        else:
            # fallthrough to system report
            flg_success = 0

    if not flg_success:
        # System report (same style as install_channelattributionpro)
        import shutil, re
        from typing import Optional, Dict, Any

        def _read_os_release() -> Optional[dict]:
            path = "/etc/os-release"
            if not os.path.exists(path):
                return None
            data = {}
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or "=" not in line or line.startswith("#"):
                        continue
                    k, v = line.split("=", 1)
                    data[k] = v.strip().strip('"').strip("'")
            return data

        def _linux_distro_fallback() -> Optional[str]:
            info = _read_os_release()
            if not info:
                return None
            return info.get("PRETTY_NAME") or " ".join(
                x for x in [info.get("NAME"), info.get("VERSION")] if x
            )

        def _which_compiler() -> Optional[str]:
            for exe in ("gcc", "cc", "clang"):
                if shutil.which(exe):
                    return exe
            return None

        def _compiler_version(exe: str) -> Optional[str]:
            try:
                p = subprocess.run([exe, "-dumpfullversion"], capture_output=True, text=True)
                if p.returncode == 0 and p.stdout.strip():
                    return f"{exe} {p.stdout.strip()}"
            except Exception:
                pass
            try:
                p = subprocess.run([exe, "--version"], capture_output=True, text=True)
                if p.returncode == 0 and p.stdout:
                    first = p.stdout.splitlines()[0].strip()
                    m = re.search(r"(gcc|clang)[^0-9]*([0-9]+(?:\.[0-9]+){0,3})", first, re.I)
                    return f"{m.group(1).lower()} {m.group(2)}" if m else first
            except Exception:
                pass
            return None

        def get_system_info(as_json: bool = False) -> Dict[str, Any] | str:
            system = platform.system()
            release = platform.release()
            arch = platform.machine() or platform.processor() or "unknown"
            py_impl = platform.python_implementation()
            py_ver = platform.python_version()

            distro_str = None
            if system == "Linux":
                try:
                    import distro  # type: ignore
                    name = distro.name(pretty=True) or distro.id() or ""
                    vers = distro.version(best=True) or ""
                    distro_str = " ".join(x for x in (name, vers) if x).strip() or None
                except Exception:
                    distro_str = _linux_distro_fallback()
            elif system == "Darwin":
                try:
                    p = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True)
                    if p.returncode == 0:
                        distro_str = f"macOS {p.stdout.strip()}"
                except Exception:
                    pass
            elif system == "Windows":
                distro_str = f"Windows {platform.release()} (build {platform.version()})"

            comp = _which_compiler()
            comp_ver = _compiler_version(comp) if comp else None

            info = {
                "os": system,
                "os_release": release,
                "architecture": arch,
                "distro": distro_str,
                "python_implementation": py_impl,
                "python_version": py_ver,
                "compiler": comp_ver or "not found",
            }
            return json.dumps(info, indent=2) if as_json else info

        print("Installation failed. Send the following information:")
        print()
        print(get_system_info())
        print()
        print("to info@channelattribution.io.")
        # just return (no exceptions)
        return

    # success path already printed; return quietly
    return
