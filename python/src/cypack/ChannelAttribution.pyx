# ChannelAttribution: Markov model for online multi-channel attribution
# Copyright (C) 2015 -   Davide Altomare and David Loris <https://channelattribution.io>

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

__version="2.2.4"
print("*** Looking to run more advanced attribution? Install ChannelAttribution Pro for free running install_pro(). Visit https://channelattribution.io for more info.")
print("Version: " + str(__version))

__message_pro="*** Install ChannelAttribution Pro for free running install_pro(). Visit https://channelattribution.io for more info. Set flg_pro=False to hide this message."

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
    
def heuristic_models(Data,var_path,var_conv,var_value=None, sep=">", flg_pro=True):

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
    flg_pro : bool, default True
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
    
    if flg_pro==True:
        print(__message_pro)

    return(res)

    
def choose_order(Data,var_path,var_conv,var_null,max_order=10,sep=">",ncore=1,roc_npt=100,plot=True, flg_pro=True):

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
    flg_pro : bool, default True
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
    
    if flg_pro==True:
        print(__message_pro)
    
    return(res_auc,res_roc,best_order)
            
    
        
def markov_model(Data,var_path,var_conv,var_value=None,var_null=None,order=1,nsim_start=1e5,max_step=None,out_more=False,sep=">",ncore=1, nfold=10, seed=0, conv_par=0.05,rate_step_sim=1.5,verbose=True, flg_pro=True):

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
    flg_pro : bool, default True
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
    
    if flg_pro==True:
        print(__message_pro)
    
    return(res)
    

def transition_matrix(Data,var_path,var_conv,var_null,order=1,sep=">",flg_equal=True, flg_pro=True):

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
    flg_pro : bool, default True
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
    
    if flg_pro==True:
        print(__message_pro)

    return(res)
    
    
    
def auto_markov_model(Data, var_path, var_conv, var_null, var_value=None, max_order=10, roc_npt=100, plot=False, nsim_start=1e5, max_step=None, out_more=False, sep=">", ncore=1, nfold=10, seed=0, conv_par=0.05, rate_step_sim=1.5, verbose=True, flg_pro=True):

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
    flg_pro : bool, default True
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
            
    [res_auc,res_roc,best_order] = choose_order(Data, var_path, var_conv, var_null, max_order = max_order, sep = sep, ncore = ncore, roc_npt = roc_npt, plot = plot, flg_pro=False)
    
    res = markov_model(Data, var_path, var_conv, var_value = var_value, var_null = var_null, order = best_order, nsim_start = nsim_start, max_step = max_step, out_more = out_more, sep = sep, ncore = ncore, nfold = nfold, seed = seed, conv_par = conv_par, rate_step_sim = rate_step_sim, verbose = verbose, flg_pro=False)
    
    if flg_pro==True:
        print(__message_pro)
    
    return(res)
