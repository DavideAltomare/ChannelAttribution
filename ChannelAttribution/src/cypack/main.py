import numpy as np
import pandas as pd
import pandas as pd
import ChannelAttribution as ca
import time


Data=pd.read_csv("C:/Users/a458057/Google Drive/Projects/ChannelAttribution/Python/ChannelAttribution-1.18.0/src/cypack/data/Data.csv",sep=";")

start_time = time.time()
ca.markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", conv_par=0.01)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ca.markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", conv_par=0.01, ncore=2, out_more=True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ca.markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value", order=3, conv_par=0.01, ncore=2, out_more=True)
print("--- %s seconds ---" % (time.time() - start_time))

data(PathData)
Data[5:5]
res=markov_model_mp(Data[5:6], "path", "total_conversions", var_value="total_conversion_value", order=3, conv_par=0.01, ncore=2, out_more=1)




var_path="path"
var_conv="total_conversions"
var_value="total_conversion_value"
var_null="total_null"

res0=transition_matrix(Data, var_path="path", var_conv="total_conversions",var_null="total_null", order=1, sep=">", flg_equal=True)

res=dict()
res['channels']=pd.DataFrame({'id_channel':range(1,len(res0[0][2])+1), 'channel_name':pd.Series(res0[0][2]).str.decode('utf-8')})
res['transition_matrix']=pd.DataFrame({'channel_from':pd.Series(res0[0][0]).str.decode('utf-8'),'channel_to':pd.Series(res0[0][1]).str.decode('utf-8'),'transition_probability':res0[1]})




max_order=10;sep=">";ncore=1;roc_npt=100;plot=True

order=1;nsim_start=1e5;max_step=None;out_more=False;sep=">";ncore=1; nfold=10; seed=0; conv_par=0.05;rate_step_sim=1.5;verbose=True


markov_model(Data, "path", "total_conversions")


#heuristic models

heuristic_models(Data,"path","total_conversions")
heuristic_models(Data,"path","total_conversions",var_value="total_conversion_value")

#markov models

markov_model(Data, "path", "total_conversions")
markov_model(Data, "path", "total_conversions","total_conversion_value")
markov_model(Data,"path","total_conversions", out_more=True)
markov_model(Data, "path", "total_conversions", var_value="total_conversion_value",out_more=True)

#markov_model_mp

markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value")
markov_model_mp(Data,"path","total_conversions", var_value="total_conversion_value",var_null="total_null")
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value",var_null="total_null", out_more=True)
markov_model_mp(Data, "path", "total_conversions", var_value="total_conversion_value",var_null="total_null", out_more=True, ncore=2, order=3)

###

var_path="path"
var_conv="total_conversions"
var_value="total_conversion_value"
var_null="total_null"

max_order=10;sep=">";ncore=1;roc_npt=100;plot=True

order=1;nsim_start=1e5;max_step=None;out_more=False;sep=">";ncore=1; nfold=10; seed=0; conv_par=0.05;rate_step_sim=1.5;verbose=True


var_path="path"
var_conv="total_conversions"
var_value="total_conversion_value"
var_null="total_null"













##########
#ESEMPIO 1
##########

#rolling autocorrelation python

start_time = timeit.default_timer()

def lagcorr(xx, lag):
	x = np.array(xx[lag:])
	y = np.array(xx[:-lag])
	indx = (~(np.isnan(x))) & (~(np.isnan(y)))
	x1 = x[indx]
	y1 = y[indx]

	return stats.pearsonr(x1,y1)[0]

y=Data['A'].rolling(window=window0+lag0,min_periods=min_periods0).apply(lambda z : lagcorr(z,lag=lag0))
elapsed_1 = timeit.default_timer() - start_time

#stats.pearsonr(Data['A'].values[999000:1000000],Data['A'].values[998999:999999])[0]

#rolling autocorrelation cython

start_time = timeit.default_timer()
rolling_acf_Data(Data=Data,cols=['A'],lag=lag0,window=window0,n_non_miss=min_periods0) 
elapsed_2 = timeit.default_timer() - start_time

#comparison

print("Rolling Acf with cython is " + str(int(elapsed_1/elapsed_2)) + " faster than Rolling Acf with python") 

# #ESEMPIO DATI WIND
# Data=pd.read_csv("/deva/Wind/G1-17_features_v2.csv",sep=";")
# rolling_acf_Data(Data=Data,cols=['AMB_WINDDIR_ABS_AVG','AMB_WINDSPEED_AVG'],lag=10,window=1000,n_non_miss=300) 
# Data


# start_time = timeit.default_timer()
# rolling_mean_Data(Data=Data,cols=['A'],window=1000,n_non_miss=300) 
# elapsed_1 = timeit.default_timer() - start_time

# #np.mean(Data['A'].values[999000:1000000])

start_time = timeit.default_timer()
x=Data['A'].rolling(window=1000, min_periods=300).mean()
elapsed_2 = timeit.default_timer() - start_time

start_time = timeit.default_timer()
x=Data['A'].rolling(window=1000, min_periods=300).apply(lambda x: np.nanmedian(x)).values
elapsed_3 = timeit.default_timer() - start_time

tmp.rolling(window=first,min_periods=int(first*missrat)).apply(mad).values


##########
#ESEMPIO 2
##########

x=[0,1,2,3,3,3,3,3,3,3,3,3,9,10,10,10,10,10,10,10,11,12,13]
frozen(x,4)

##########
#ESEMPIO 3
##########


x=[1,2,3,4,5]
y=ci(x)
y[0]

###############
#ESEMPIO TStore
###############

#creazione dati esempio
n1=13e6
n2=13e5

Data=pd.DataFrame({'var1':np.arange(int(n1)),'var2': np.arange(int(n1))})
Data['var2']=Data['var2']+0.111
Data.index=np.arange(10e6,10e6+len(Data)*2,2)
Data.index=pd.to_datetime(Data.index,unit='s')
Data1=Data.iloc[0:(int(n1)-int(n2))]
Data2=Data.iloc[(int(1e6)+(int(n1)-int(n2))):(int(1e6)+(int(n1)-int(n2))+int(1e6))]
Data3=pd.DataFrame({'var3': np.arange(int(n1))})
Data4=Data1.iloc[4356789:6234567]
del Data4['var1']
Data4['var2']=0.999
del Data

#creo oggetto di tipo TStore  
dir='/devd/tmp/Prova/'			
ts=TStore(dir)
ts.clean()

#riempo oggetto con dati da un data.frame
ts.from_Data(Data=Data1,vname=['var1','var2'],vtype=['float32','float64'],vtime_start=Data1.index[0],vmb_part=5,freq_sec=2,vpad=1)
# a=np.memmap('/devd/tmp/Prova/var1/p9.bin',dtype='float32',mode='r')
# x=pd.Series(a[:].tolist())
# x[x==-9223372036854775808]=np.nan

#ottengo informazioni sui dati
ts.read_info()

#importo dati dall'oggetto a un data.frame
ts.to_Data(select_time=[pd.to_datetime('1970-12-01 00:00:01'),pd.to_datetime('1970-12-31 23:59:59')],select_col=['var1','var2'],join_type='inner',index_int=False)
#ts.to_Data(select_time=[pd.to_datetime('1971-01-22 13:46:38'),pd.to_datetime('1971-01-22 13:46:38')],select_col=['var1','var2'],join_type='outer',index_int=False)
#ts.to_Data(select_time=[pd.to_datetime('1971-01-22 13:46:36'),pd.to_datetime('1971-01-22 13:47:30')],select_col=['var1','var2'],join_type='outer',index_int=False)
#ts.to_Data(select_time=[pd.to_datetime('1971-01-22 13:46:40'),pd.to_datetime('1971-01-22 13:47:30')],select_col=['var1','var2'],join_type='outer',index_int=False)

#accodo righe di un data.frame all'oggetto
ts.append_Data(Data=Data2,vname=['var1'],vtime_start=Data2.index[0])
ts.read_info()
ts.to_Data(select_col=['var1','var2'],join_type='outer',index_int=False)
# ts.to_Data(select_time=[pd.to_datetime('1971-01-22 13:46:36'),pd.to_datetime('1971-02-14 17:20:00')],select_col=['var1','var2'],join_type='outer',index_int=False)
# ts.to_Data(select_time=[pd.to_datetime('1971-02-21 15:59:44'),pd.to_datetime('1971-02-21 16:00:00')],select_col=['var1','var2'],join_type='outer',index_int=False)

#modifico righe dell'oggetto sulla base dei valori assunti da un data.frame 
ts.update_Data(Data4,select_col=['var2'])
ts.to_Data(select_time=[pd.to_datetime('1970-08-05 14:12:50'),pd.to_datetime('1970-09-18 01:25:40')],select_col=['var2'],index_int=False)

#aggiungo colonne di un data.frame all'oggetto
ts.from_Data(Data3,vname=['var3'],vtype=['float64'],vtime_start=pd.to_datetime('1980-03-01 00:00:00'),vmb_part=5,freq_sec=2,vpad=1) 
ts.read_info()
ts.to_Data(join_type='outer',index_int=False)

#rinomino colonne
ts.rename_cols(['var3'],['z'])
ts.read_info()

#rimuovo colonne dall'oggetto
ts.del_cols(cols=['z']) 
ts.read_info()

#apply_row
def fmedian(x):
	return(pd.Series.median(x,axis=1,skipna=True))
	
ts.apply_row(func=fmedian,index_int=False)

#apply_col (SPERIMENTALE)
def fmedian(x):
	return(pd.Series.median(x,axis=0,skipna=True))


Datatmp=ts.apply_col(func=fmedian,cols=None,select_time=None,ncore=2)

#altro esempio apply row/col
Data=pd.DataFrame(np.random.randn(int(10e6),10), columns=list('ABCDEFGHIL'))

dir='/devd/tmp/'
ts=TStore(dir)
ts.clean()
ts.from_Data(Data=Data,vname=list(Data.columns),vtype='float32',vtime_start=0,vmb_part=10,freq_sec=2)
x=ts.to_Data(index_int=False)

def fmedian(x):
	return(pd.Series.median(x,axis=1,skipna=True))
	
y=ts.apply_row(func=fmedian,index_int=False)

str(str(y.iloc[78654].values[0])+ '....'+ str(np.median(x.iloc[78654])))

def fmedian(x):
	return(pd.Series.median(x,axis=0,skipna=True))

y=ts.apply_col(func=fmedian,cols=None,select_time=None,ncore=2)

# ##########
# #ESEMPIO 2
# ##########

# #rolling autocorrelation python

# start_time = timeit.default_timer()
# y=Data['A'].rolling(window=window0,min_periods=min_periods0).median()
# elapsed_1 = timeit.default_timer() - start_time

# #rolling median cython

# start_time = timeit.default_timer()
# rolling_median_Data(Data=Data,cols=['A'],window=window0,n_non_miss=min_periods0) 
# elapsed_2 = timeit.default_timer() - start_time