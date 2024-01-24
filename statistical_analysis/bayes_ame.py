#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:15:49 2024

@author: elizabethespinosa
"""

### Imports
import numpy as np
import pandas as pd

""""""" ******** POOLED MODELS ***********"""""""

""" Function that takes observations (DataFrame DF), trace, variable labels 'selvars', 
    posterior labels 'varspost', and incremental scale 'h'.
    Returns dataframe with Average Marginal Effects """

def ame_pooled(DF, trace, selvars, varspost, h):
    DFame= pd.DataFrame()
    dictRes= posterior_dictionary_pooled(trace, varspost) #obtain posterior means
    dfpost= pd.DataFrame.from_dict(dictRes['mean'])
    dfpost.rename(columns= {0: 'Pooled'}, inplace=True)
    dfpost.index= selvars

    dfmarg= average_marginal_effects_pooled(DF,dfpost, h, selvars)

    dfmarg.rename(columns= {0: 'AME', 1: 'std', 2: 'CI25', 3: 'CI97'}, inplace=True)
    dfmarg.index= selvars
    return dfmarg

""" Function that takes trace and variables X (names) and 
    Returns posterior means summary for each parameter in a dictionary """
def posterior_dictionary_pooled(trace, varspost):
    dictR = {}
    
    for x in varspost:
        meanarr= np.hstack(trace['posterior'][x]).mean(axis=0) #mean posterior
        cib= np.percentile(np.hstack(trace['posterior'][x]), 2.5, axis=0) # lower bound Credible Interval
        cit= np.percentile(np.hstack(trace['posterior'][x]), 97.5, axis=0)# Upper bound Credible Interval
        for arr, label in zip([meanarr, cib, cit], ['mean','cib','cit']):
            dictR.setdefault(label, []).append(arr)

    return dictR

""" Function that computes Average Marginal Effects for pooled data """

def average_marginal_effects_pooled(DF, dfpost, h, selvars):
    DF['cons']= 1
    xbhat_names= []
    ame_list= []
    std_list= []
    percentile025_list= []
    percentile97_list = []
    
    dfame= DF
    
    for x in selvars: #for each k variable 
        
        # Add columns with beta estimates 
        dfame['beta_'+x]= dfpost.loc[(dfpost.index==x)].sum()[0]
        # generate columns with X*B:
        dfame[x+'_hat']= dfame['beta_'+x]*dfame[x]
        xbhat_names.append(x+'_hat') 
    
    originalist= xbhat_names
    for x in selvars:
        dfame=  calculate_phat(dfame, xbhat_names, 'yhat') # Obtain Y0 estimated with variables in actual state
        newlist= []
        newlist = [n for n in originalist if n != x+'_hat'] #remove x from list (keep rest)
        newlist.append(x+'_wave') #append new variable instead
        
        # Discrete variables Marginal Effect
        
        if x== 'year_2017' in x: #discrete variables change from 0 to 1 
            # Compute yhat with discrete variable as zero

            xbhat_names2= xbhat_names #new list to remove X zero values
            xbhat_names2.remove(x+'_hat')
            dfame=  calculate_phat(dfame, xbhat_names2, 'yhat')#yhat with x as zero
            # Compute yhat with discrete varible as 1
            dfame[x+'_wave']= 1* dfame['beta_'+x]
            dfame= calculate_phat(dfame, newlist, 'y_wave')  # y_wave (with x=1)
            dfame[x+'_me']= (dfame['y_wave']- dfame['yhat'])
            
            ame_list.append(dfame[x+'_me'].mean()) # Average dy/dx
            std_list.append(dfame[x+'_me'].std())
            percentile025_list.append(np.percentile(dfame[x+'_me'], 2.5))
            percentile97_list.append(np.percentile(dfame[x+'_me'], 97.5))
        
            
        # Constant (don't need marginal effects)
        elif x== 'cons':
            dfame[x+'_me']= np.exp(dfame['beta_'+x])/(1+np.exp(dfame['beta_'+x])) 
            ame_list.append(dfame[x+'_me'].mean()) # Average dy/dx
            std_list.append(dfame[x+'_me'].std())
            percentile025_list.append(np.percentile(dfame[x+'_me'], 2.5))
            percentile97_list.append(np.percentile(dfame[x+'_me'], 97.5))
            
        # Continuous variables
        else:
            dfame[x+'_wave']= (dfame[x]+h)*dfame['beta_'+x] # new variable X(1+h)*B
            dfame= calculate_phat(dfame, newlist, 'y_wave')  # y_wave (y with X+h)
            dfame[x+'_me']= (dfame['y_wave']- dfame['yhat'])/h #dy/dx
            
            ame_list.append(dfame[x+'_me'].mean()) # Average dy/dx
            std_list.append(dfame[x+'_me'].std())
            percentile025_list.append(np.percentile(dfame[x+'_me'], 2.5))
            percentile97_list.append(np.percentile(dfame[x+'_me'], 97.5))
        
    return pd.DataFrame([ame_list,std_list, percentile025_list,percentile97_list]).T

""""""" ******** MULTILEVEL MODELS ***********"""""""
""" Function that takes observational DataFrame DF, trace, variable labels 'selvars', 
    posterior labels 'varspost', and incremental scale 'h'.
    Returns dataframe with Average Marginal Effects for each farm class """

def ame_multilevel(DF, trace, selvars, varspost, h, multilevelvar, nlevels, listlabels):
    DFame= pd.DataFrame()
    dictRes= posterior_dictionary_multilevel(trace, varspost) #obtain posterior means for classes
    dfpost= pd.DataFrame.from_dict(dictRes['mean'])
    for x,z in zip(listlabels, np.arange(0, len(listlabels)).tolist()):
        dfpost.rename(columns= {z: x}, inplace=True)
    dfpost.index= selvars

    for cl in listlabels: #for each class group
        # Obtain Average Marginal Effects

        dfmarg= average_marginal_effects(DF,dfpost, cl, h, selvars, multilevelvar)

        dfmarg[multilevelvar]= cl
        DFame= DFame.append(dfmarg) #append AME results for each class

    DFame.rename(columns= {0: 'AME', 1: 'std', 2: 'CI25', 3: 'CI97'}, inplace=True)
    DFame.index= selvars*nlevels
    return DFame

""" Function that takes trace and variables X (names) and 
    Returns posterior means summary for each parameter in a dictionary """
    
def posterior_dictionary_multilevel(trace, varspost):
    dictR = {}
    
    for x in varspost:
        meanarr= np.vstack(trace['posterior'][x]).mean(axis=0) #mean posterior
        cib= np.percentile(np.vstack(trace['posterior'][x]), 2.5, axis=0) # lower bound Credible Interval
        cit= np.percentile(np.vstack(trace['posterior'][x]), 97.5, axis=0)# Upper bound Credible Interval
        for arr, label in zip([meanarr, cib, cit], ['mean','cib','cit']):
            dictR.setdefault(label, []).append(arr)

    return dictR
    
""" Function to compute Averge Marginal Effects for multilevel data"""

def average_marginal_effects(DF, dfpost, classize, h, selvars, multilevelvar):
    DF['cons']= 1
    xbhat_names= []
    ame_list= []
    std_list= []
    percentile025_list= []
    percentile97_list = []
    
    dfame= DF.loc[DF[multilevelvar]== classize] # truncate data to farm class
    
    for x in selvars: #for each k variable 
        
        # Add columns with beta estimates 
        dfame['beta_'+x]= dfpost[classize].loc[(dfpost.index==x)].sum()
        # generate columns with X*B:
        dfame[x+'_hat']= dfame['beta_'+x]*dfame[x]
        xbhat_names.append(x+'_hat') 
        
    # Obtain Y0 estimated with variables in actual state
    dfame=  calculate_phat(dfame, xbhat_names, 'yhat')
    
    originalist= xbhat_names
    for x in selvars:
        dfame=  calculate_phat(dfame, xbhat_names, 'yhat')
        newlist= []
        newlist = [n for n in originalist if n != x+'_hat'] #remove x from list (keep rest)
        newlist.append(x+'_wave') #append new variable instead
        
        # Discrete variables Marginal Effect
        
        if x== 'year_2017' in x: #discrete variables change from 0 to 1 
            # Compute yhat with discrete variable as zero

            xbhat_names2= xbhat_names #new list to remove X zero values
            xbhat_names2.remove(x+'_hat')
            dfame=  calculate_phat(dfame, xbhat_names2, 'yhat')
            # Compute yhat with discrete varible as 1
            dfame[x+'_wave']= 1* dfame['beta_'+x]
            dfame= calculate_phat(dfame, newlist, 'y_wave')  # y_wave (y with X+h)
            dfame[x+'_me']= (dfame['y_wave']- dfame['yhat'])
        
            
        # Constant (don't need marginal effects)
        elif x== 'cons':
            dfame[x+'_me']= np.exp(dfame['beta_'+x])/(1+np.exp(dfame['beta_'+x])) 
            
        # Continuous variables
        else:
            dfame[x+'_wave']= (dfame[x]+h)*dfame['beta_'+x] # new variable X(1+h)*B
            dfame= calculate_phat(dfame, newlist, 'y_wave')  # y_wave (y with X+h)
            dfame[x+'_me']= (dfame['y_wave']- dfame['yhat'])/h #dy/dx

        ame_list.append(dfame[x+'_me'].mean()) # Average dy/dx
        std_list.append(dfame[x+'_me'].std())
        percentile025_list.append(np.percentile(dfame[x+'_me'], 2.5))
        percentile97_list.append(np.percentile(dfame[x+'_me'], 97.5))
        
    return pd.DataFrame([ame_list,std_list, percentile025_list,percentile97_list]).T

""" COMMON FUNCTIONS"""




""" Function to compute p= e^xb/(1+e^xb) given data 'df', variables X 'xvars', result's 'name'"""

def calculate_phat(df, xvars, name):
    df['xbhat_sum']= df[xvars].sum(axis=1)
    df[name]= np.exp(df['xbhat_sum'])/(1+np.exp(df['xbhat_sum']))
    return df

