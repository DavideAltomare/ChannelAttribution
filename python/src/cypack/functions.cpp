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

#include "functions.h"

using namespace std;
using namespace arma;
 

pair < vector<string>,list< vector<double> > > heuristic_models_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, string sep)
{

	 
 //inp.b
 
 bool flg_var_value;
 flg_var_value=1;
 if((unsigned long int) vv.size()==0){
  flg_var_value=0;
 } 
	 
 unsigned long int i,j,k,lvy,ssize;
 bool cfirst;
 unsigned long int start_pos,end_pos;
 unsigned long int nchannels;
 string s,channel,channel_first,channel_last;
  
 lvy=(unsigned long int) vy.size();
 nchannels=0;
 
 map<string,unsigned long int> mp_channels;
 vector<string> vchannels;
	  	
 map<string,double> mp_first_conv;
 map<string,double> mp_first_val;	
 map<string,double> mp_last_conv;
 map<string,double> mp_last_val;
 map<string,double> mp_linear_conv;
 map<string,double> mp_linear_val;
 map<string,double> mp0_linear_conv;
 map<string,double> mp0_linear_val;

 vector<string> vchannels_unique;
 double nchannels_unique;
 string kchannel;
 unsigned long int n_path_length;

 for(i=0;i<lvy;i++){
	 	 
  s=vy[i];
  
  s+=sep[0];
  ssize=(unsigned long int) s.size();
  channel="";
  j=0;
  nchannels_unique=0;
  vchannels_unique.clear();
  
  n_path_length=0;
  mp0_linear_conv.clear();
  mp0_linear_val.clear();
  
  start_pos=0;
  end_pos=0;  
  
  while(j<ssize){  
         
   cfirst=1;
   while(s[j]!=sep[0]){
	if(cfirst==0){   
     if(s[j]!=' '){
	  end_pos=j;	 
	 }
    }else if((cfirst==1) & (s[j]!=' ')){
	 cfirst=0;
	 start_pos=j;
	 end_pos=j;
	}
    ++j;     
   }
   
   if(cfirst==0){
    channel=s.substr(start_pos,(end_pos-start_pos+1));
   
    if(mp_channels.find(channel) == mp_channels.end()){
	 mp_channels[channel]=nchannels;
	 vchannels.push_back(channel);
	 ++nchannels;
	
     mp_first_conv[channel]=0;
	 mp_last_conv[channel]=0;
	 mp_linear_conv[channel]=0;
	 mp0_linear_conv[channel]=0;
	 
	 if(flg_var_value==1){
	  mp_first_val[channel]=0;	
	  mp_last_val[channel]=0;
	  mp_linear_val[channel]=0;
	  mp0_linear_val[channel]=0;
	 }		 
	 
	}
	 	 
    //list unique channels
    if(nchannels_unique==0){
     vchannels_unique.push_back(channel);
	 ++nchannels_unique;
    }else if(find(vchannels_unique.begin(),vchannels_unique.end(),channel)==vchannels_unique.end()){
	 vchannels_unique.push_back(channel);
	 ++nchannels_unique;
    }

 	mp0_linear_conv[channel]=mp0_linear_conv[channel]+vc[i];
    if(flg_var_value==1){
	 mp0_linear_val[channel]=mp0_linear_val[channel]+vv[i]; 
    }
	++n_path_length;
   
    channel_last=channel;
  
   }//end cfirst
  
   channel="";
   ++j;
    
  }//end while j
   
  channel_first=vchannels_unique[0];
  mp_first_conv[channel_first]=mp_first_conv[channel_first]+vc[i];
 
  mp_last_conv[channel_last]=mp_last_conv[channel_last]+vc[i];
 
  //linear
  for(k=0;k<nchannels_unique;k++){
    kchannel=vchannels_unique[k];
    mp_linear_conv[kchannel]=mp_linear_conv[kchannel]+(mp0_linear_conv[kchannel]/n_path_length);
  }
  
  if(flg_var_value==1){
   mp_first_val[channel_first]=mp_first_val[channel_first]+vv[i];   
   mp_last_val[channel_last]=mp_last_val[channel_last]+vv[i];  
   for(k=0;k<nchannels_unique;k++){
    kchannel=vchannels_unique[k];
    mp_linear_val[kchannel]=mp_linear_val[kchannel]+(mp0_linear_val[kchannel]/n_path_length); 
   }
  }	  
 
  
 }//end for i
 
 vector<double> vfirst_conv(nchannels);
 vector<double> vlast_conv(nchannels);
 vector<double> vlinear_conv(nchannels); 
 
 vector<double> vfirst_val(nchannels);
 vector<double> vlast_val(nchannels);
 vector<double> vlinear_val(nchannels);
 
 for(k=0;k<nchannels;k++){
  kchannel=vchannels[k];	 
  vfirst_conv[k]=mp_first_conv[kchannel];
  vlast_conv[k]=mp_last_conv[kchannel];
  vlinear_conv[k]=mp_linear_conv[kchannel];
  
  if(flg_var_value==1){
   vfirst_val[k]=mp_first_val[kchannel];
   vlast_val[k]=mp_last_val[kchannel];
   vlinear_val[k]=mp_linear_val[kchannel];
  }
  
 }
  
 list< vector<double> > res1;
 
 res1.push_back(vfirst_conv);
 res1.push_back(vlast_conv); 
 res1.push_back(vlinear_conv);
 
 if(flg_var_value==1){
  res1.push_back(vfirst_val);
  res1.push_back(vlast_val); 
  res1.push_back(vlinear_val);	 
 }
 
 pair <vector<string>,list< vector<double> > > res;
 
 res = make_pair(vchannels,res1);
 
 return(res);
    
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Distribution function Class

class Fx
{
 
 SpMat<unsigned long int> S;
 SpMat<unsigned long int> S0;
 SpMat<unsigned long int> S1;
 vector<unsigned long int> lrS0;
 vector<unsigned long int> lrS; 
 unsigned long int non_zeros,nrows,val0,lval0,i,j,k,s0,lrs0i;
 
 public:
  Fx(unsigned long int nrow0,unsigned long int ncol0): S(nrow0,ncol0), S0(nrow0,ncol0), S1(nrow0,ncol0), lrS0(nrow0,0), lrS(nrow0,0), non_zeros(0), nrows(nrow0) {}
  void init(unsigned long int, unsigned long int);
  void add(unsigned long int, unsigned long int,unsigned long int);
  void cum();
  unsigned long int sim(unsigned long int, double);
  double pconv(unsigned long int, unsigned long int);   
  pair < list< vector<string> >, vector<double> > tran_matx(vector<string>);      
};  

void Fx::init(unsigned long int nrow1, unsigned long int ncol1)
{
  S.reset();
  S.set_size(nrow1,ncol1);
  
  S0.reset();
  S0.set_size(nrow1,ncol1);
  
  S1.reset();
  S1.set_size(nrow1,ncol1);  
  
  lrS0.clear();
  lrS0.resize(nrow1);
  
  lrS.clear();
  lrS.resize(nrow1);
  
  non_zeros=0;
  nrows=nrow1;
} 


void Fx::add(unsigned long int ichannel_old, unsigned long int ichannel, unsigned long int vxi)
{
    
  val0=S(ichannel_old,ichannel); //fill d.f. transition with vxi
  if(val0==0){
   lval0=lrS0[ichannel_old];
   S0(ichannel_old,lval0)=ichannel;
   lrS0[ichannel_old]=lval0+1;
   ++non_zeros;
  }
  S(ichannel_old,ichannel)=val0+vxi; 
  
} 

void Fx::cum()
{

 for(i=0;i<nrows;i++){
  lrs0i=lrS0[i];
  if(lrs0i>0){
   S1(i,0)=S(i,S0(i,0));
   for(j=1;j<lrs0i;j++){
    S1(i,j)=S1(i,j-1)+S(i,S0(i,j));
   }
   lrS[i]=S1(i,lrs0i-1);
  }   
 }    
  
}


unsigned long int Fx::sim(unsigned long int c, double uni) 
{
  
 s0=(unsigned long int) floor(uni*lrS[c]+1);
 
 for(k=0; k<lrS0[c]; k++){   
  if(S1(c,k)>=s0){return(S0(c,k));}
 }

 return 0;
    
}


pair < list< vector<string> >, vector<double> > Fx::tran_matx(vector<string> vchannels) 
{

 unsigned long int mij,sm3;
 vector<string> vM1(non_zeros);
 vector<string> vM2(non_zeros);
 vector<double> vM3(non_zeros);
 vector<double> vsm;
 vector<unsigned long int> vk;
 
 k=0;
 for(i=0;i<nrows;i++){
  sm3=0;
  for(j=0;j<lrS0[i];j++){
   mij=S(i,S0(i,j));
   if(mij>0){   
      vM1[k]=vchannels[i];
      vM2[k]=vchannels[S0(i,j)];
      vM3[k]=mij;
      sm3=sm3+mij;
      ++k;
    }
  }
  
  vsm.push_back(sm3);
  vk.push_back(k);
  
 }//end for
 
 unsigned long int w=0;
 for(k=0;k<non_zeros;k++){
  if(k==vk[w]){++w;}
  vM3[k]=vM3[k]/vsm[w]; 
 }
   
 list< vector<string> > res1;

 res1.push_back(vM1);
 res1.push_back(vM2);
  
 pair < list< vector<string> >, vector<double> > res;
 
 res = make_pair(res1,vM3);
 
 return(res);
 
}


double Fx::pconv(unsigned long int ichannel, unsigned long int nchannels)
{

 double res=0;
 for(k=(nchannels-2);k<nchannels;k++){ //consider only conversion channel and null channel
  res=res+(double) S(ichannel,k);
 }
 
 if(res>0){
  res=(double) S(ichannel,nchannels-2)/res;
 }else{
  res=0;	 
 }
 
 return(res);
  
} 



vector<long int> split_string(const string &s, unsigned long int order) {
    
	char delim=' ';
	vector<long int> result(order,-1);
    stringstream ss (s);
    string item;

	unsigned long int h=0;
    while (getline (ss, item, delim)) {
		result[h]=stoi(item);
		h=h+1;
    }
		
    return result;
}



// void print(auto &input)
// {
	// for (unsigned long int i = 0; i < (unsigned long int) input.size(); i++) {
		// std::cout << input.at(i) << ' ';
	// }
// }


vector<unsigned long int> bounds(unsigned long int parts, unsigned long int mem) {
    vector<unsigned long int>bnd;
    unsigned long int delta = mem / parts;
    unsigned long int reminder = mem % parts;
    unsigned long int N1 = 0, N2 = 0;
    bnd.push_back(N1);
    for (unsigned long int i = 0; i < parts; ++i) {
        N2 = N1 + delta;
        if (i == parts - 1)
            N2 += reminder;
        bnd.push_back(N2);
        N1 = N2;
    }
    return bnd;
}


void W_choose_order_1(vector<string> vy, unsigned long int lvy, vector<unsigned long int> vc, vector<unsigned long int> vn, unsigned long int roc_npt, unsigned long int nchannels, vector<unsigned long int> &vorder, vector<double> &vuroc, vector<double> &vuroc_corr, list < vector<double> > &L_roc, unsigned long int from_W, unsigned long int to_W)
{


 for(unsigned long int order = (from_W+1); order < (to_W+1); order++){
  
  string s,channel,path;
  unsigned long int nchannels_sim,i,ssize,ichannel_old,j,nc,start_pos,end_pos,start_pos_last,ichannel,npassi,vci,vni,vpi,k,h;  
  vector<string> vchannels_sim;
  map<string,unsigned long int> mp_channels_sim; 
  vector<string> vy2(lvy);  
  bool flg_next_path,flg_next_null;
  vector<double> vprev(lvy);
  double min_prev,max_prev,pauc,nnodes;
  
  vector<double> vth(roc_npt);
  unsigned long int tp,fn,tn,fp;
  double th,tpr,fpr,tpr_old,fpr_old,auc;

  vector<unsigned long int> vlastc(lvy);  
  
  //output
  
  vector<double> vtpr(roc_npt+1);
  vector<double> vfpr(roc_npt+1);
  
  nnodes=exp(lgamma(nchannels-3+order-1+1)-lgamma(order+1)-lgamma(nchannels-3-1+1));
  
  unsigned long int np=0;
  for(i=0;i<lvy; i++){
   np=np+vc[i]+vn[i];
  }
  
  if(nnodes<np){
  
   nchannels_sim=0;
    
   mp_channels_sim["(start)"]=0;
   vchannels_sim.push_back("(start)");
   ++nchannels_sim;
        
   for(i=0;i<lvy;i++){
   
    s=vy[i];
    ssize=(unsigned long int) s.size();
   
    channel="";
    ichannel_old=0;
    path="";
    j=0; 
    nc=0;
    start_pos=0;
    end_pos=0;
    start_pos_last=0; 
       
    while(j<ssize){ 
     
     while((j<ssize) & (nc<order)){
      
  	  if(s[j]==' '){
  	   nc=nc+1;
  	   if(nc==1){
  	    start_pos=j;
  	   }
  	  }else{
  	   end_pos=j;	
  	  }
  	
      ++j;
     
     } //while(s[j]!=' ') 
     
     channel=s.substr(start_pos_last,(end_pos-start_pos_last+1));
     
     if(mp_channels_sim.find(channel) == mp_channels_sim.end()){
      mp_channels_sim[channel]=nchannels_sim;
      vchannels_sim.push_back(channel);
      ++nchannels_sim;
     }
     
     ichannel=mp_channels_sim[channel];
     
     vlastc[i]=ichannel;
     
     if(ichannel!=ichannel_old){
      path+=to_string(ichannel);
      path+=" ";
     }   
     
     ichannel_old=ichannel;
   
     if((end_pos+1)==ssize){break;}; 
     j=start_pos+1;
     start_pos_last=j;   
     nc=0;
   
    }//end while(j<ssize)
   
    vy2[i]="0 "+path+"e";
  	
   }//end for i
   
   mp_channels_sim["(conversion)"]=nchannels_sim;
   vchannels_sim.push_back("(conversion)");    
   ++nchannels_sim;
    
   mp_channels_sim["(null)"]=nchannels_sim;
   vchannels_sim.push_back("(null)");     
   ++nchannels_sim;
      
   //Compute transition matrix
     
   npassi=0;
   Fx S(nchannels_sim,nchannels_sim);  
   
   for(i=0;i<lvy;i++){
      
    flg_next_path=0;
    flg_next_null=0;
                             
    s=vy2[i];
    s+=" ";
    ssize= (unsigned long int) s.size();
   
    channel="";
    ichannel_old=0;
    ichannel=0;
   
    j=0;
    npassi=0;
   
    vci=vc[i];
    vni=vn[i];
         
    vpi=vci+vni;
    
    while((j<ssize) & (flg_next_path==0)){
       
     while(s[j]!=' '){
   
      if(j<ssize){
       channel+=s[j];
      }
      j=j+1;
     }
                
     if(channel[0]!='0'){//if it is not channel start
     
      if(channel[0]=='e'){ //end state
      
       ++npassi;
       
       if(vci>0){ //if there are conversions
        ichannel=nchannels_sim-2;
	    S.add(ichannel_old,ichannel,vci);
       
        if(vni>0){
         flg_next_null=1;
        }else{
         flg_next_path=1;
        }
               
       }
      
       if(((vni>0) | (flg_next_null==1)) & (flg_next_path==0)){ //if there are not conversions
        ichannel=nchannels_sim-1;
        S.add(ichannel_old,ichannel,vni);
      
        flg_next_path=1;
       }
      
      }else{ //not end state
       
       if(vpi>0){
        ichannel=atol(channel.c_str());
	    S.add(ichannel_old,ichannel,vpi);
	   }
     
      }
     
      if(flg_next_path==0){
       ++npassi;
      }
     }else{ //starting state
    
      ichannel=0;
    
     }
   
     if(flg_next_path==0){
      ichannel_old=ichannel;
     }
      
     if(flg_next_path==0){
      channel="";
      j=j+1;   
     }
    
    }//end while j<size
       
   }//end for
           
   //fit
    
   for(i=0;i<lvy;i++){	   
    vprev[i]=S.pconv(vlastc[i],nchannels_sim);  
   } 
   
   min_prev=1;
   for(i=0;i<lvy;i++){	   
    min_prev=min(min_prev,vprev[i]);  
   } 
   
   max_prev=0;
   for(i=0;i<lvy;i++){	   
    max_prev=max(max_prev,vprev[i]);  
   } 
     
   for(k=0;k<roc_npt;k++){
    vth[k]=min_prev+(k*(max_prev-min_prev)/(roc_npt-1));	
   }
     
   auc=0; 
   tpr_old=0;
   fpr_old=0;
      
   vtpr[0]=0;
   vfpr[0]=0;  
   h=1;	
	 
   for(k=(roc_npt-1);k>=0 && k<roc_npt;k--){	   
     
    tp=0,fn=0,tn=0,fp=0;   
    th=vth[k];
    
    for(i=0;i<lvy;i++){
     
     if((vprev[i]>=th) & (vc[i]>0)){
  	 tp=tp+vc[i];   
     }else if((vprev[i]<th) & (vc[i]>0)){
      fn=fn+vc[i];
     }
     
     vni=vn[i];
	 
	 if((vprev[i]<th) & (vni>0)){
  	 tn=tn+vni;   
     }else if((vprev[i]>=th) & (vni>0)){
  	 fp=fp+vni;   
     }
       
    }
   
    tpr=(double)tp/(double)(tp+fn);
    fpr=(double)fp/(double)(fp+tn);
    
    auc=auc+((fpr-fpr_old)*tpr_old)+(((fpr-fpr_old)*(tpr-tpr_old))/2);
       
    vtpr[h]=tpr;
    vfpr[h]=fpr;
    
    tpr_old=tpr;
    fpr_old=fpr;
    
    h=h+1;
    
   }//end for k
   
   vtpr[roc_npt]=1;
   vfpr[roc_npt]=1; 
   auc=auc+((1-fpr_old)*tpr_old)+(((1-fpr_old)*(1-tpr_old))/2);
       
   pauc=(double)(1-((1-auc)*((np-1)/(np-nnodes-1))));
   if((pauc<0) | (pauc>1)){
    pauc=0;	  
   }
   
   vuroc[order-1]=auc;
   vuroc_corr[order-1]=pauc;
   vorder[order-1]=order;
   
   L_roc.push_back(vfpr);
   L_roc.push_back(vtpr);
     
  }//end if(nnodes<lvy)
 
 }
  
}


list< vector<double> > choose_order_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int max_order, string sep, unsigned long int ncore, unsigned long int roc_npt)
{
 

 //inp.b 
      
 unsigned long int i,j,lvy,ssize;
 unsigned long int nchannels,npassi;
 bool cfirst;
 unsigned long int start_pos,end_pos;
 string s,channel,path;
 map<string,unsigned long int> mp_channels,mp_channels_sim;
 map<unsigned long int,unsigned long int> mp_npassi;
 vector<unsigned long int> vnpassi;
    
 lvy=(unsigned long int) vy.size();
   
 ////////////////////////
 //ENCODING FROM ONE STEP 
 ////////////////////////
      
 string channel_test="";
 string channel_old;
  
 nchannels=0;
   
 mp_channels["(start)"]=0;
 vector<string> vchannels;
 vchannels.push_back("(start)");     
 ++nchannels;
    
 for(i=0;i<lvy;i++){
	        
  s=vy[i];
  s+=sep[0];
  ssize=(unsigned long int) s.size();
  channel="";
  path="";
  j=0;
  npassi=0;
  start_pos=0;
  end_pos=0;
   
  //channel.touch
  
  while(j<ssize){  
         
   cfirst=1;
   while(s[j]!=sep[0]){
    if(cfirst==0){   
     if(s[j]!=' '){
      end_pos=j;     
     }
    }else if((cfirst==1) & (s[j]!=' ')){
     cfirst=0;
     start_pos=j;
     end_pos=j;
    }
    ++j;     
   }
   
   if(cfirst==0){
    channel=s.substr(start_pos,(end_pos-start_pos+1));
   
    if(mp_channels.find(channel) == mp_channels.end()){
     mp_channels[channel]=nchannels;
     vchannels.push_back(channel);
     ++nchannels;
    }
			 
    if(npassi==0){
	 path="";
    }else{
     path+=" ";
    }
     
    path+=to_string(mp_channels[channel]);
    ++npassi;      
           
   }//if end_pos
   
   channel="";
   ++j;
   
  }//end while channel
    
  vy[i]=path;
  ++npassi;
  
 }//end for
  
 mp_channels["(conversion)"]=nchannels;
 vchannels.push_back("(conversion)");    
 ++nchannels;
  
 mp_channels["(null)"]=nchannels;
 vchannels.push_back("(null)");     
 ++nchannels;
 
 //turn to order=1
 
 list < vector<double> > L_roc;
 vector<double> vuroc(max_order);
 vector<double> vuroc_corr(max_order);
 vector<unsigned long int> vorder(max_order);
 
 vector<unsigned long int> limits = bounds(ncore, max_order);
  
 if(ncore==1){
  
  W_choose_order_1(vy, lvy, vc, vn, roc_npt, nchannels, ref(vorder), ref(vuroc), ref(vuroc_corr), ref(L_roc), limits[0], limits[1]);	  
 
 }else{
  
  vector<thread> threads(ncore);

  //Launch ncore threads:	
  for(unsigned long int td=0; td<ncore; td++){
   threads[td]=thread(W_choose_order_1, vy, lvy, vc, vn, roc_npt, nchannels, ref(vorder), ref(vuroc), ref(vuroc_corr), ref(L_roc), limits[td], limits[td+1]);
  }

  //Join the threads with the main thread
  for(auto &t : threads){
   t.join();
  }  
  
 }
 
 vector<double> vorder2(vorder.begin(), vorder.end());
 
 L_roc.push_back(vorder2);
 L_roc.push_back(vuroc);
 L_roc.push_back(vuroc_corr);
   
 return(L_roc);
  
}




void W_markov_model_1(unsigned long int seed, vector<double> v_vui, unsigned long int lvy, vector<unsigned long int> vc, unsigned long int nch0, vector<double> vv, unsigned long int nfold, unsigned long int nsim_start,  map< unsigned long int, vector<long int> > mp_channels_sim_inv, unsigned long int max_npassi, unsigned long int nchannels_sim, unsigned long int order, bool flg_var_value, unsigned long int nchannels, Fx S, Fx fV, vector<unsigned long int> &nconv, vector<double> &ssval, vector< vector<double> > &T, vector< vector<double> > &TV, vector< vector<double> > &V, vector< vector<double> > &VV, vector<double> &v_inc_path, unsigned long int from_W, unsigned long int to_W)
{
 
 long int id0;
 unsigned long int i0,c,npassi0,k0,c_last=0,n_inc_path;
 vector<bool> C(nchannels,0);
 double sval0=0,sn,sm;
 bool flg_exit;
 
 for(unsigned long int run = from_W; run < to_W; run++){
	 
   mt19937 generator(seed+run);  
   uniform_real_distribution<double> distribution(0,1);
   
   n_inc_path=0;   

   for(i0=0; i0<(unsigned long int) (nsim_start/nfold); i0++){
	          	        
     c=0;
     npassi0=0;
 	    
     for(k0=0; k0<nchannels; k0++){ //set vector with visited channels to 0
      C[k0]=0;
     }
    
     C[c]=1; //set channel start to 1
     
 	 flg_exit=0;
 	
     while((npassi0<=max_npassi) & (flg_exit==0)){ //stop if maxium number of steps is reached
      
      c=S.sim(c,distribution(generator));
     
      if(c==nchannels_sim-2){ //stop if conversion is reached
       flg_exit=1;
      }else if(c==nchannels_sim-1){ //stop if null is reached
       flg_exit=1;
      }
      
      if(flg_exit==0){
        if(order==1){
          C[c]=1; //set to 1 visited channel   
        }else{       
         for(k0=0; k0<order; k0++){
		   id0=mp_channels_sim_inv[c][k0];
           if(id0>=0){
            C[id0]=1;
           }else{
            break;     
           }
          }
        }
         
        c_last=c; //store visited channel
        ++npassi0;
      }
    	
     }//end while npassi0 
 	
     
     if(c==nchannels_sim-2){ //only if conversion is reached visited channels will be set to 1 (if maximum number of channel is reached it is like null is reached)
      
     nconv[run]=nconv[run]+1; //increase conversions
      
     //generate for channel c_last a conversion value "sval0"
     if(flg_var_value==1){
      sval0=v_vui[fV.sim(c_last,distribution(generator))];
     }   
         
     ssval[run]=ssval[run]+sval0;
       
     for (k0=0; k0<nchannels; k0++){
       if(C[k0]==1){
         T[run][k0]=T[run][k0]+1;
        if(flg_var_value==1){
          V[run][k0]=V[run][k0]+sval0;
        }
       }
      }
    
     }else if(c!=nchannels_sim-1){ //if also NULL has not been reached
	   
       n_inc_path=n_inc_path+1;	   
		 	 
	 }//end if conv
           
   }//end for i
   
   v_inc_path[run]=(double) n_inc_path/ (double) i0;
       
   T[run][0]=0; //set channel start = 0
   T[run][nchannels-2]=0; //set channel conversion = 0 
   T[run][nchannels-1]=0; //set channel null = 0 
   
   sn=0; 
   for(k0=0;k0<lvy; k0++){
    sn=sn+vc[k0];
   }
    
   sm=0;
   for(k0=0;k0<nchannels-1; k0++){
    sm=sm+T[run][k0];
   }
   
   for (k0=1; k0<(nch0+1); k0++){
    if(sm>0){
     TV[run][k0-1]=(T[run][k0]/sm)*sn;
    }
   }
     
   if(flg_var_value==1){
  
    V[run][0]=0; //set channel start = 0
    V[run][nchannels-2]=0; //set channel conversion = 0 
    V[run][nchannels-1]=0; //set channel null = 0 
      
    sn=0;
    for(k0=0;k0<lvy; k0++){
     sn=sn+vv[k0];
    }
    
    sm=0;
    for(k0=0;k0<nchannels-1; k0++){
     sm=sm+V[run][k0];
    }
      
    for(k0=1; k0<(nch0+1); k0++){
     if(sm>0){
      VV[run][k0-1]=(V[run][k0]/sm)*sn;
     }
    }
     
   }
		
 }//end for run

}

string f_print_perc(double num){ 
 
 string res;
 if(num>=1){
  res=to_string((double)(floor(num*10000)/100)).substr(0,6);    
 }else if(num>=0.1){ 
  res=to_string((double)(floor(num*10000)/100)).substr(0,5); 
 }else{
  res=to_string((double)(floor(num*10000)/100)).substr(0,4);    	   
 } 
 return(res);
}



pair < list< vector<string> >,list< vector<double> > > markov_model_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<double>& vv, vector<unsigned long int>& vn, unsigned long int order, unsigned long int nsim_start,  unsigned long int max_step, unsigned long int out_more, string sep, unsigned long int ncore, unsigned long int nfold, unsigned long int seed, double conv_par, double rate_step_sim, int verbose)
{
 
 //inp.b 
    
 bool flg_var_value;
 flg_var_value=1;
 if((unsigned long int) vv.size()==0){
  flg_var_value=0;
 }
 
 bool flg_var_null;
 flg_var_null=1;
 if((unsigned long int) vn.size()==0){
  flg_var_null=0;
 }
  
 unsigned long int i,j,k,lvy,ssize;
 unsigned long int nchannels,nchannels_sim,npassi;
 bool cfirst;
 unsigned long int start_pos,end_pos;
 string s,channel,path;
 map<string,unsigned long int> mp_channels,mp_channels_sim;
 map< unsigned long int, vector<long int> > mp_channels_sim_inv;
 map<unsigned long int,unsigned long int> mp_npassi;
 vector<unsigned long int> vnpassi;
    
 lvy=(unsigned long int) vy.size();
   
 ////////////////////////
 //ENCODING FROM ONE STEP 
 ////////////////////////
    
 //conversion value map
 unsigned long int l_vui=0;
 map<double,unsigned long int> mp_vui;
 vector<double> v_vui;
 double vui;

 vector<string> rchannels;
 string channel_j;
  
 nchannels=0;
 nchannels_sim=0;
 
 vector<string> vy2(lvy);
 
 mp_channels["(start)"]=0;
 vector<string> vchannels;
 vchannels.push_back("(start)");     
 ++nchannels;

 vector<string> vchannels_sim;

 //conversion value map definition
 if(flg_var_value==1){
  for(i=0;i<lvy;i++){
   if(vc[i]>0){
    vui=vv[i]/vc[i];
    if(mp_vui.find(vui)==mp_vui.end()){
     mp_vui[vui]=l_vui;
     v_vui.push_back(vui);
     ++l_vui;    
    }
   }
  }
 }
 
 //encoding channels name to integer
   
 for(i=0;i<lvy;i++){
    
  s=vy[i];
  s+=sep[0];
  ssize=(unsigned long int) s.size();
  channel="";
  path="";
  j=0;
  npassi=0;
  start_pos=0;
  end_pos=0;
     
  //channel.touch
  
  while(j<ssize){  
         
   cfirst=1;
   while(s[j]!=sep[0]){ 
    if(cfirst==0){   
     if(s[j]!=' '){
      end_pos=j;     
     }
    }else if((cfirst==1) & (s[j]!=' ')){
     cfirst=0;
     start_pos=j;
     end_pos=j;
    }
    ++j;     
   }
   
   if(cfirst==0){
    channel=s.substr(start_pos,(end_pos-start_pos+1));
   
    if(mp_channels.find(channel) == mp_channels.end()){
     mp_channels[channel]=nchannels;
     vchannels.push_back(channel);
     ++nchannels;
    }
			 
    if(npassi==0){
	 path="";
    }else{
     path+=" ";
    }
     
    path+=to_string(mp_channels[channel]);
    ++npassi;      
           
   }//if end_pos
   
   channel="";
   ++j;
   
  }//end while channel
    
  vy[i]=path;
  ++npassi;
 
 }//end for

 mp_channels["(conversion)"]=nchannels;
 vchannels.push_back("(conversion)");    
 ++nchannels;
  
 mp_channels["(null)"]=nchannels;
 vchannels.push_back("(null)");     
 ++nchannels;
 
 //turn to order=1
    	
 nchannels_sim=0;
   
 mp_channels_sim["(start)"]=0;
 vchannels_sim.push_back("(start)");
 ++nchannels_sim;
   
 unsigned long int ichannel,ichannel_old,vpi,vci,vni; 
 bool flg_next_path,flg_next_null;
 unsigned long int start_pos_last,nc;
 
 vector<long int> vtmp(order);
   
 for(i=0;i<lvy;i++){
  
   s=vy[i];
   ssize=(unsigned long int) s.size();
  
   channel="";
   ichannel_old=0;
   path="";
   j=0; 
   nc=0;
   start_pos=0;
   end_pos=0;
   start_pos_last=0; 
      
   while(j<ssize){ 
    
    while((j<ssize) & (nc<order)){
     
 	if(s[j]==' '){
 	 nc=nc+1;
 	 if(nc==1){
 	  start_pos=j;
 	 }
 	}else{
 	 end_pos=j;	
 	}
 	
     ++j;
    
    } //while(s[j]!=' ') 
    
    channel=s.substr(start_pos_last,(end_pos-start_pos_last+1));
    	
    if(mp_channels_sim.find(channel) == mp_channels_sim.end()){
     mp_channels_sim[channel]=nchannels_sim;
	 
	 vtmp=split_string(channel,order);
	 mp_channels_sim_inv[nchannels_sim]=vtmp;
    
	 vchannels_sim.push_back(channel);
     ++nchannels_sim;
    }
    
    ichannel=mp_channels_sim[channel];
        
    if(ichannel!=ichannel_old){
     path+=to_string(ichannel);
     path+=" ";
    }   
    
    ichannel_old=ichannel;

    if((end_pos+1)==ssize){break;}; 
    j=start_pos+1;
    start_pos_last=j;   
    nc=0;
  
   }//end while(j<ssize)
  
   vy2[i]="0 "+path+"e";
 	
 }//end for i
  
 mp_channels_sim["(conversion)"]=nchannels_sim;
 vchannels_sim.push_back("(conversion)");    
 ++nchannels_sim;
   
 mp_channels_sim["(null)"]=nchannels_sim;
 vchannels_sim.push_back("(null)");     
 ++nchannels_sim;
     
 //////////////////////////////
 //BUILDING SIMULATION MATRICES
 //////////////////////////////
  
 string channel_old;
  
 Fx S(nchannels_sim,nchannels_sim);
  
 Fx fV(nchannels_sim,l_vui);
 
 unsigned long int max_npassi; 
 max_npassi=0; 
 
 for(i=0;i<lvy;i++){
     
  flg_next_path=0;
  flg_next_null=0;
                            
  s=vy2[i];
  s+=" ";
  ssize= (unsigned long int) s.size();
  
  channel="";
  ichannel_old=0;
  ichannel=0;
 
  j=0;
  npassi=0;
  
  vci=vc[i];
  if(flg_var_null==1){
   vni=vn[i];
  }else{
   vni=0;
  }      
  vpi=vci+vni;
   
  while((j<ssize) & (flg_next_path==0)){
      
   while(s[j]!=' '){
  
    if(j<ssize){
     channel+=s[j];
    }
    j=j+1;
   }
               
   if(channel[0]!='0'){//if it is not channel start
   
    if(channel[0]=='e'){ //end state
    
     ++npassi;
     
     if(vci>0){ //if there are conversions
      ichannel=nchannels_sim-2;
      S.add(ichannel_old,ichannel,vci);
    
      if(flg_var_value==1){
       vui=vv[i]/vci;
	   fV.add(ichannel_old,mp_vui[vui],vci);
      }

      if(vni>0){
       flg_next_null=1;  
      }else{
       flg_next_path=1;
      }
             
     }
    
     if(((vni>0) | (flg_next_null==1)) & (flg_next_path==0)){ //iif there are not conversions
      ichannel=nchannels_sim-1;
      S.add(ichannel_old,ichannel,vni);
      flg_next_path=1;
     }
    
    }else{ //end state
     
     if(vpi>0){
      ichannel=atol(channel.c_str());
      S.add(ichannel_old,ichannel,vpi);
     }
   
    }
   
    if(flg_next_path==0){
     ++npassi;
    }
   }else{ //end state
   
    ichannel=0;
   
   }
  
   if(flg_next_path==0){
    ichannel_old=ichannel;
   }
     
   if(flg_next_path==0){
    channel="";
    j=j+1;   
   }
   
  }//end while j<size
   
  max_npassi=max(max_npassi,npassi);
   
 }//end for 
 
 //out transition matrix
 
 pair < list< vector<string> >, vector<double> > res_mtx; 
 
 if(out_more==1){
   res_mtx=S.tran_matx(vchannels_sim);
 }
   
 //distribution function for transition
 S.cum(); 
    
 //distribution function conversion value
 if(flg_var_value==1){
  fV.cum();
 }
    
 //SIMULATIONS
     
 if(max_step>0){
  max_npassi=max_step;
 }
   
 if(nsim_start==0){
  nsim_start=(unsigned long int) 100000;
 }
       
 unsigned long int run;
 unsigned long int nch0; 
 double sn=0;
  
 vector< vector<double> > T(nfold,vector<double>(nchannels,0));
 vector< vector<double> > V(nfold,vector<double>(nchannels,0));
 
 nch0=nchannels-3;

 vector< vector<double> > TV(nfold,vector<double>(nch0,0));
 vector<double> rTV(nch0,0);
 
 vector< vector<double> > VV(nfold,vector<double>(nch0,0));
 vector<double> rVV(nch0,0);
 
 vector<unsigned long int> nconv(nfold,0);
 vector<double> ssval(nfold,0);
 
 vector<double> TV_fin(nch0);
 vector<double> VV_fin(nch0);
 vector<double> vtmp1(nfold);
 vector<double> v_res_conv(nfold);
 
 vector<double> v_inc_path(nfold);
    
 double max_res_conv=numeric_limits<double>::infinity();
 double min_res_conv;
 double res_conv;
 
 unsigned long int id_mz,run_min_res_conv=0;
 
 string pout; 
  
 while(max_res_conv>conv_par){
	
  min_res_conv=numeric_limits<double>::infinity(); 
	
  vector<unsigned long int> limits = bounds(ncore, nfold);

  if(ncore==1){
	  
	W_markov_model_1(seed, v_vui, lvy, vc, nch0, vv, nfold, nsim_start, mp_channels_sim_inv, max_npassi, nchannels_sim, order, flg_var_value, nchannels, S, fV, ref(nconv), ref(ssval), ref(T), ref(TV), ref(V), ref(VV), ref(v_inc_path), limits[0], limits[1]);
	  
  }else{
  
   vector<thread> threads(ncore);
      
   //Launch ncore threads:	  
   for(unsigned long int td=0; td<ncore; td++){
    threads[td]=thread(W_markov_model_1, seed, v_vui, lvy, vc, nch0, vv, nfold, nsim_start, mp_channels_sim_inv, max_npassi, nchannels_sim, order, flg_var_value, nchannels, S, fV, ref(nconv), ref(ssval), ref(T), ref(TV), ref(V), ref(VV), ref(v_inc_path), limits[td], limits[td+1]);
   }
      
   //Join the threads with the main thread
   for(auto &t : threads){
  	t.join();
   } 
	    
  }
  
  sn=0; 
  for(k=0;k<lvy; k++){
   sn=sn+vc[k];
  }
   
  for(k=0; k<nch0; k++){ 
   for(run=0; run<nfold; run++){
    vtmp1[run]=TV[run][k];
   }
   sort(vtmp1.begin(), vtmp1.end(), std::greater<double>());
   id_mz=(unsigned long int) (nfold/2);
   if(nfold % 2 == 0){
	TV_fin[k]=(vtmp1[id_mz-1]+vtmp1[id_mz])/2;   
   }else{
	TV_fin[k]=vtmp1[id_mz];   
   }
  }
   
  if(flg_var_value==1){ 

   for(k=0; k<nch0; k++){ 
    for(run=0; run<nfold; run++){
     vtmp1[run]=VV[run][k];
    }
    sort(vtmp1.begin(), vtmp1.end(), std::greater<double>());
    id_mz=(unsigned long int) (nfold/2);
    if(nfold % 2 == 0){
	 VV_fin[k]=(vtmp1[id_mz-1]+vtmp1[id_mz])/2;   
    }else{
	 VV_fin[k]=vtmp1[id_mz];   
    }
   }  
  
  } 
     
  max_res_conv=0;
  for(run=0; run<nfold; run++){
	  
   res_conv=0;
   for(k=0; k<nch0; k++){
    res_conv=res_conv+abs(TV[run][k]-TV_fin[k]);
   }
   res_conv=res_conv/sn;
   if(res_conv>max_res_conv){
    max_res_conv=res_conv;
   }
   if(res_conv<min_res_conv){
    min_res_conv=res_conv;
	run_min_res_conv=run;
   }
   
  }
      
  if(verbose==1){
   if(max_res_conv>conv_par){    
	pout="print('Number of simulations: "+ to_string(nsim_start) + " - Reaching convergence (wait...): " + f_print_perc(max_res_conv) + "% > " + f_print_perc(conv_par) + "%')";
	PyRun_SimpleString(pout.c_str());
   }else{
	pout="print('Number of simulations: "+ to_string(nsim_start) + " - Convergence reached: " + f_print_perc(max_res_conv) + "% < " + f_print_perc(conv_par) + "%')";
	PyRun_SimpleString(pout.c_str());
   }
  }
  
  nsim_start=(unsigned long int) (nsim_start*rate_step_sim);

 }//end while(res_conv>conv_par)
     
 vector<string> vchannels0(nch0);
 for(k=1; k<(nch0+1); k++){
  vchannels0[k-1]=vchannels[k];
 }
  
 double succ_path=0;
 for(k=0; k<nfold; k++){
  succ_path=succ_path+(1-v_inc_path[k])/nfold;
 }	  
 if(verbose==1){ 
  pout="print('Percentage of simulated paths that successfully end before maximum number of steps (" + to_string(max_npassi) + ") is reached: " + f_print_perc(succ_path) + "%')";
  PyRun_SimpleString(pout.c_str());
 }
 
 
 list< vector<double> > res1;
 list< vector<string> > res2;
 
 res1.push_back(TV[run_min_res_conv]);
 if(out_more==1){
 //removal effects conversion	
  for(k=1; k<(nch0+1); k++){
   rTV[k-1]=T[run_min_res_conv][k]/nconv[run_min_res_conv];
  } 
 
  res1.push_back(rTV);
  res1.push_back(res_mtx.second); 
  auto it = next(res_mtx.first.begin(), 0);
  res2.push_back(*it);
  it = next(res_mtx.first.begin(), 1);  
  res2.push_back(*it);
 }
 
 if(flg_var_value==1){
  res1.push_back(VV[run_min_res_conv]);
  if(out_more==1){
   //removal effects conversion value	
   for(k=1; k<(nch0+1); k++){
    rVV[k-1]=V[run_min_res_conv][k]/ssval[run_min_res_conv];
   } 
   res1.push_back(rVV);
  }
 }

 res2.push_back(vchannels0);
 
 pair < list< vector<string> >,list< vector<double> > > res;
 
 res = make_pair(res2,res1);
 
 return(res);

}	


pair < list< vector<string> >, vector<double> > transition_matrix_cpp(vector<string>& vy, vector<unsigned long int>& vc, vector<unsigned long int>& vn, unsigned long int order, string sep, int flg_equal)
{
	
 //inp.b 
     
 bool flg_var_null;
 flg_var_null=1;
 if((unsigned long int) vn.size()==0){
  flg_var_null=0;
 }

  
 unsigned long int i,j,lvy,ssize;
 unsigned long int nchannels,nchannels_sim,npassi;
 bool cfirst;
 unsigned long int start_pos,end_pos;
 string s,channel,path;
 map<string,unsigned long int> mp_channels,mp_channels_sim;
 map< unsigned long int, vector<long int> > mp_channels_sim_inv;
 map<unsigned long int,unsigned long int> mp_npassi;
 vector<unsigned long int> vnpassi;
    
 lvy=(unsigned long int) vy.size();
   
 ////////////////////////
 //ENCODING FROM ONE STEP 
 ////////////////////////
    
 //conersion value map
 map<double,unsigned long int> mp_vui;
 vector<double> v_vui;

 vector<string> rchannels;
 string channel_j;
  
 nchannels=0;
 nchannels_sim=0;
 
 vector<string> vy2(lvy);
 
 mp_channels["(start)"]=0;
 vector<string> vchannels;
 vchannels.push_back("(start)");     
 ++nchannels;

 vector<string> vchannels_sim;
 
 //encoding channel names to integers
  
 for(i=0;i<lvy;i++){
       
  s=vy[i];
  s+=sep[0];
  ssize=(unsigned long int) s.size();
  channel="";
  path="";
  j=0;
  npassi=0;
  start_pos=0;
  end_pos=0;
   
  //channel.touch
  
  while(j<ssize){  
         
   cfirst=1;
   while(s[j]!=sep[0]){ 
    if(cfirst==0){   
     if(s[j]!=' '){
      end_pos=j;     
     }
    }else if((cfirst==1) & (s[j]!=' ')){
     cfirst=0;
     start_pos=j;
     end_pos=j;
    }
    ++j;     
   }
   
   if(cfirst==0){
    channel=s.substr(start_pos,(end_pos-start_pos+1));
   
    if(mp_channels.find(channel) == mp_channels.end()){
     mp_channels[channel]=nchannels;
     vchannels.push_back(channel);
     ++nchannels;
    }
			 
    if(npassi==0){
	 path="";
    }else{
     path+=" ";
    }
     
    path+=to_string(mp_channels[channel]);
    ++npassi;      
           
   }//if end_pos
   
   channel="";
   ++j;
   
  }//end while channel
    
  vy[i]=path;
  ++npassi;
 
 }//end for

 mp_channels["(conversion)"]=nchannels;
 vchannels.push_back("(conversion)");    
 ++nchannels;
  
 mp_channels["(null)"]=nchannels;
 vchannels.push_back("(null)");     
 ++nchannels;
 
 //turn order=1
    
 nchannels_sim=0;
   
 mp_channels_sim["(start)"]=0;
 vchannels_sim.push_back("(start)");
 ++nchannels_sim;
   
 unsigned long int ichannel,ichannel_old,vpi,vci,vni; 
 bool flg_next_path,flg_next_null;
 unsigned long int start_pos_last,nc;
 
 vector<long int> vtmp(order);

 for(i=0;i<lvy;i++){ 
   
   s=vy[i];
   
   ssize=(unsigned long int) s.size();
  
   channel="";
   ichannel_old=0;
   path="";
   j=0; 
   nc=0;
   start_pos=0;
   end_pos=0;
   start_pos_last=0; 
      
   while(j<ssize){ 
    
    while((j<ssize) & (nc<order)){
     
 	if(s[j]==' '){
 	 nc=nc+1;
 	 if(nc==1){
 	  start_pos=j;
 	 }
 	}else{
 	 end_pos=j;	
 	}
 	
     ++j;
    
    } //while(s[j]!=' ') 
    
    channel=s.substr(start_pos_last,(end_pos-start_pos_last+1));
    	
    if(mp_channels_sim.find(channel) == mp_channels_sim.end()){
     mp_channels_sim[channel]=nchannels_sim;
	 
	 vtmp=split_string(channel,order);
	 mp_channels_sim_inv[nchannels_sim]=vtmp;
    
	 vchannels_sim.push_back(channel);
     ++nchannels_sim;
    }
    
    ichannel=mp_channels_sim[channel];
        
	if(flg_equal==0){	
     if(ichannel!=ichannel_old){
      path+=to_string(ichannel);
      path+=" ";
	  ichannel_old=ichannel;
     }   
    }else{
     path+=to_string(ichannel);
     path+=" ";	
	}
    
    if((end_pos+1)==ssize){break;}; 
    j=start_pos+1;
    start_pos_last=j;   
    nc=0;
  
   }//end while(j<ssize)
  
   vy2[i]="0 "+path+"e";
   
 }//end for i
  
 mp_channels_sim["(conversion)"]=nchannels_sim;
 vchannels_sim.push_back("(conversion)");    
 ++nchannels_sim;
   
 mp_channels_sim["(null)"]=nchannels_sim;
 vchannels_sim.push_back("(null)");     
 ++nchannels_sim;
  
 //////////////////////////////
 //BUILDING SIMULATION MATRICES
 //////////////////////////////
 
 string channel_old;
  
 Fx S(nchannels_sim,nchannels_sim);
     
 for(i=0;i<lvy;i++){
     
  flg_next_path=0;
  flg_next_null=0;
                            
  s=vy2[i];
  s+=" ";
  ssize= (unsigned long int) s.size();
  
  channel="";
  ichannel_old=0;
  ichannel=0;
 
  j=0;
  npassi=0;
  
  vci=vc[i];
  if(flg_var_null==1){
   vni=vn[i];
  }else{
   vni=0;
  }      
  vpi=vci+vni;
   
  while((j<ssize) & (flg_next_path==0)){
      
   while(s[j]!=' '){
  
    if(j<ssize){
     channel+=s[j];
    }
    j=j+1;
   }
               
   if(channel[0]!='0'){//if it is not channel start
    
     if(channel[0]=='e'){ //end state
     
      ++npassi;
      
      if(vci>0){ //if there are conversions
       ichannel=nchannels_sim-2;
	   S.add(ichannel_old,ichannel,vci);
    
       if(vni>0){
        flg_next_null=1;  
       }else{
        flg_next_path=1;
       }
              
      }
     
      if(((vni>0) | (flg_next_null==1)) & (flg_next_path==0)){ //if there are not conversions
       ichannel=nchannels_sim-1;
       S.add(ichannel_old,ichannel,vni);
       flg_next_path=1;
      }
     
     }else{ //end state
      
      if(vpi>0){
       ichannel=atol(channel.c_str());
	   S.add(ichannel_old,ichannel,vpi);
	  }
    
     }
    
     if(flg_next_path==0){
      ++npassi;
     }
    }else{ //starting state
   
     ichannel=0;
   
    }
  
    if(flg_next_path==0){
     ichannel_old=ichannel;
    }
     
   if(flg_next_path==0){
    channel="";
    j=j+1;   
   }
   
  }//end while j<size
      
 }//end for 
 
 unsigned long int k,nch0;
 nch0=nchannels-3;
 
 vector<string> vchannels0(nch0);
 for(k=1; k<(nch0+1); k++){
  vchannels0[k-1]=vchannels[k];
 }
 
 
 pair < list< vector<string> >, vector<double> > res; 
 res=S.tran_matx(vchannels_sim);
 res.first.push_back(vchannels0);
   
 return(res);
 
} 	
