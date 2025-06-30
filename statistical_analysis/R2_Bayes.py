#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:29:55 2025

@author: elizabethespinosa
"""

import numpy as np

""" Function to Compute Bayes R2 for pooled or hierarchical/multilevel models """
""" Takes data (DF), linkfunction ('logit' or 'identity'), trace, observed outcome 'yvar', 
    selected vars from dataframe 'selvas',
    variable labels from posterior estimates 'varpost',
    boolean var 'hierarchical= TRUE/FALSE'
    multilevel variable in dataframe 'multilevelvar' {0, 1, 2, ...},
    number of chains 'chainum',
    number of draws per chain 'ndraws',
    number of levels 'nlevels',
     """
    
def Bayesian_Rsquared(DF, linkfunction, trace, yvar,selvars, varspost, hierarchical, multilevelvar, chainum, ndraws, nlevels):

    Rsquared= []
    DF['cons']= 1

    for c in range(chainum): #for c in chains
        for d in range(ndraws): #for d in draws
            Ywave= []
            Yres= []
            if hierarchical== True:
                for f in range(nlevels): 
                    df= DF[DF[multilevelvar]== f]
                    Xmatrix= df[selvars].to_numpy()   ## get matrix X of covariates
                    Yarray= df[yvar].to_numpy()   ## get observed Y
                    betas = []
                    for x in varspost: #for x in selected variables
                        betas_array= np.hstack(trace['posterior'][x][c][d]) #get vector of parameters
                        betas.append(betas_array[f])
                    betas= np.asarray(betas) 
                    XB= np.dot(Xmatrix, betas.T)
                    if linkfunction== 'logit':
                        ypred= np.exp(XB)/(1+np.exp(XB))
                    elif linkfunction== 'identity':
                        ypred= XB
                    Ywave.append(ypred) #Y fitted| params
                    Yres.append(Yarray- ypred)  # Y residual
            else: 
                df= DF.copy()
                Xmatrix= df[selvars].to_numpy()   ## get matrix X of covariates
                Yarray= df[yvar].to_numpy()   ## get observed Y
                betas = []
                for x in varspost: #for x in selected variables
                    betas.append(trace['posterior'][x][c][d])#get vector of parameters
                betas= np.asarray(betas) 
                XB= np.dot(Xmatrix, betas.T)
                if linkfunction== 'logit':
                    ypred= np.exp(XB)/(1+np.exp(XB))
                elif linkfunction== 'identity':
                    ypred= XB
                
                Ywave.append(ypred) #Y fitted| params
                Yres.append(Yarray- ypred)  # Y residual


            Rsquared.append(np.var(np.hstack(Ywave))/(np.var(np.hstack(Ywave))+ np.var(np.hstack(Yres))))
    R2= np.asarray(Rsquared)
    
    return R2.mean(), np.percentile(R2, 2.5), np.percentile(R2, 97.5)
