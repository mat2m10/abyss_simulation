import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import warnings
from scipy.stats import t
from scipy import stats
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Input, Model, layers, regularizers
warnings.filterwarnings("ignore")

def no_corr(complete, y):
    Ps_no_corr = []
    snps_no_corr = []
    coefs_no_corr = []
    intercepts_no_corr = []
    faulty_snps_no_corr = []
    AFs_no_corr = []
    # create phenotype
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X = np.array(list(complete[snp]))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            snps_no_corr.append(snp.split("_AF_")[0])
            Ps_no_corr.append(p_values) 
            AFs_no_corr.append(snp.split("_AF_")[1])
            coefs_no_corr.append(beta_hat)
        except Exception as e:
            faulty_snps_no_corr.append(snp)
            print(e)
    Ps_no_corr = np.sort(Ps_no_corr)
    epsilon = 1e-100
    logPs_no_corr = np.sort(-np.log10(Ps_no_corr+epsilon))
    no_corr_df = pd.DataFrame({'coeff':coefs_no_corr, "AFs":AFs_no_corr, "Ps_no_corr": Ps_no_corr})
    n = len(no_corr_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    no_corr_df['expected_P'] = expected_quantiles
    no_corr_df['logPs_no_corr'] = logPs_no_corr
    no_corr_df['expected_logP'] = expected_logP
    return no_corr_df

def rare_pc(complete, y , PC_veryrare, rare_pc_columns):
    Ps_rare_PC = []
    snps_rare_PC = []
    coefs_rare_PC = []
    intercepts_rare_PC = []
    faulty_snps_rare_PC = []
    AFs_rare_PC = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(PC_veryrare[rare_pc_columns])))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_rare_PC.append(p_values)
            snps_rare_PC.append(snp.split("_AF_")[0])
            AFs_rare_PC.append(snp.split("_AF_")[1])
            coefs_rare_PC.append(beta_hat)
        except Exception as e:
            faulty_snps_rare_PC.append(snp)
            print(e)

    Ps_rare_PC = np.sort(Ps_rare_PC)
    epsilon = 1e-100

    logPs_rare_PCs = np.sort(-np.log10(Ps_rare_PC+epsilon))
    rare_PCs_df = pd.DataFrame({'coeff':coefs_rare_PC, "AFs":AFs_rare_PC, "Ps_rare_PCs": Ps_rare_PC})

    n = len(rare_PCs_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))

    rare_PCs_df['expected_P'] = expected_quantiles
    rare_PCs_df['logPs_rare_PCs'] = logPs_rare_PCs
    rare_PCs_df['expected_logP'] = expected_logP

    return rare_PCs_df

def pc(complete, y, PC_common, pc_columns):
    Ps_common_PC = []
    intercepts_common_PC = []
    snps_common_PC = []
    AFs_common_PC = []
    coefs_common_PC = []
    faulty_snps_common_PC = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(PC_common[pc_columns])))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_common_PC.append(p_values)
            snps_common_PC.append(snp.split("_AF_")[0])
            AFs_common_PC.append(snp.split("_AF_")[1])
            coefs_common_PC.append(beta_hat)
        except Exception as e:
            faulty_snps_common_PC.append(snp)
            print(e)

    Ps_common_PC = np.sort(Ps_common_PC)
    epsilon = 1e-100
    logPs_common_PCs = np.sort(-np.log10(Ps_common_PC+epsilon))
    common_PCs_df = pd.DataFrame({'coeff':coefs_common_PC, "AFs":AFs_common_PC, "Ps_common_PCs": Ps_common_PC})
    n = len(common_PCs_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    common_PCs_df['expected_P'] = expected_quantiles
    common_PCs_df['logPs_common_PCs'] = logPs_common_PCs
    common_PCs_df['expected_logP'] = expected_logP
    return common_PCs_df

def gc(df_no_corr):
    Ps_no_corr = np.array(df_no_corr['Ps_no_corr'])
    coefs_no_corr = list(df_no_corr['coeff'])
    AFs_no_corr = list(df_no_corr['AFs'])
    median_chi2 = np.median(stats.chi2.ppf(1 - Ps_no_corr, 1))
    expected_median = stats.chi2.ppf(0.5, 1)
    lambda_gc = median_chi2 / expected_median
    
    # Convert p-values to chi-square statistics
    chi2_values = stats.chi2.isf(Ps_no_corr, df=1)
    
    # Adjust chi-square statistics
    chi2_corr = chi2_values / lambda_gc
    
    # Convert adjusted chi-square statistics back to p-values
    p_adjusted = stats.chi2.sf(chi2_corr, df=1)
    
    Ps_GC = np.sort(p_adjusted)
    epsilon = 1e-100
    logPs_GC = np.sort(-np.log10(Ps_GC+epsilon))
    GC_df = pd.DataFrame({'coeff':coefs_no_corr, "AFs":AFs_no_corr, "Ps_GC": Ps_GC})
    n = len(GC_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    GC_df['expected_P'] = expected_quantiles
    GC_df['logPs_GC'] = logPs_GC
    GC_df['expected_logP'] = expected_logP
    return GC_df

def abyss_bottle_linreg(complete, y, AE_bottle):
    Ps_abyss_bottle = []
    intercepts_abyss_bottle = []
    snps_abyss_bottle = []
    AFs_abyss_bottle = []
    coefs_abyss_bottle = []
    faulty_snps_abyss_bottle = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(AE_bottle)))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_abyss_bottle.append(p_values)
            snps_abyss_bottle.append(snp.split("_AF_")[0])
            AFs_abyss_bottle.append(snp.split("_AF_")[1])
            coefs_abyss_bottle.append(beta_hat)
        except Exception as e:
            faulty_snps_abyss_bottle.append(snp)
            print(e)

    Ps_abyss_bottle = np.sort(Ps_abyss_bottle)
    epsilon = 1e-100
    logPs_abyss_bottle = np.sort(-np.log10(Ps_abyss_bottle+epsilon))
    abyss_bottle_df = pd.DataFrame({'coeff':coefs_abyss_bottle, "AFs":AFs_abyss_bottle, "Ps_abyss_bottle": Ps_abyss_bottle})
    n = len(abyss_bottle_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    abyss_bottle_df['expected_P'] = expected_quantiles
    abyss_bottle_df['logPs_abyss_bottle'] = logPs_abyss_bottle
    abyss_bottle_df['expected_logP'] = expected_logP
    return abyss_bottle_df

def abyss_maf_linreg(complete, y, probmaf):
    Ps_abyss_maf = []
    intercepts_abyss_maf = []
    snps_abyss_maf = []
    AFs_abyss_maf = []
    coefs_abyss_maf = []
    faulty_snps_abyss_maf = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(probmaf[snp])))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_abyss_maf.append(p_values)
            snps_abyss_maf.append(snp.split("_AF_")[0])
            AFs_abyss_maf.append(snp.split("_AF_")[1])
            coefs_abyss_maf.append(beta_hat)
        except Exception as e:
            faulty_snps_abyss_maf.append(snp)
            print(e)

    Ps_abyss_maf = np.sort(Ps_abyss_maf)
    epsilon = 1e-100
    logPs_abyss_maf = np.sort(-np.log10(Ps_abyss_maf+epsilon))
    abyss_maf_df = pd.DataFrame({'coeff':coefs_abyss_maf, "AFs":AFs_abyss_maf, "Ps_abyss_maf": Ps_abyss_maf})
    n = len(abyss_maf_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    abyss_maf_df['expected_P'] = expected_quantiles
    abyss_maf_df['logPs_abyss_maf'] = logPs_abyss_maf
    abyss_maf_df['expected_logP'] = expected_logP
    return abyss_maf_df


def deep_abyss_bottle_linreg(complete, y, deep_abyss_bottle):
    Ps_deep_abyss_bottle = []
    intercepts_deep_abyss_bottle = []
    snps_deep_abyss_bottle = []
    AFs_deep_abyss_bottle = []
    coefs_deep_abyss_bottle = []
    faulty_snps_deep_abyss_bottle = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(deep_abyss_bottle)))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_deep_abyss_bottle.append(p_values)
            snps_deep_abyss_bottle.append(snp.split("_AF_")[0])
            AFs_deep_abyss_bottle.append(snp.split("_AF_")[1])
            coefs_deep_abyss_bottle.append(beta_hat)
        except Exception as e:
            faulty_snps_deep_abyss_bottle.append(snp)
            print(e)

    Ps_deep_abyss_bottle = np.sort(Ps_deep_abyss_bottle)
    epsilon = 1e-100
    logPs_deep_abyss_bottle = np.sort(-np.log10(Ps_deep_abyss_bottle+epsilon))
    deep_abyss_bottle_df = pd.DataFrame({'coeff':coefs_deep_abyss_bottle, "AFs":AFs_deep_abyss_bottle, "Ps_deep_abyss_bottle": Ps_deep_abyss_bottle})
    n = len(deep_abyss_bottle_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    deep_abyss_bottle_df['expected_P'] = expected_quantiles
    deep_abyss_bottle_df['logPs_deep_abyss_bottle'] = logPs_deep_abyss_bottle
    deep_abyss_bottle_df['expected_logP'] = expected_logP
    return deep_abyss_bottle_df


def deep_abyss_maf_linreg(complete, y, deep_probmaf):

    Ps_deep_abyss_maf = []
    intercepts_deep_abyss_maf = []
    snps_deep_abyss_maf = []
    AFs_deep_abyss_maf = []
    coefs_deep_abyss_maf = []
    faulty_snps_deep_abyss_maf = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        X = np.column_stack((X_snp, np.array(deep_probmaf[snp])))
        X = sm.add_constant(X)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_deep_abyss_maf.append(p_values)
            snps_deep_abyss_maf.append(snp.split("_AF_")[0])
            AFs_deep_abyss_maf.append(snp.split("_AF_")[1])
            coefs_deep_abyss_maf.append(beta_hat)
        except Exception as e:
            faulty_snps_deep_abyss_maf.append(snp)
            print(e)

    Ps_deep_abyss_maf = np.sort(Ps_deep_abyss_maf)
    epsilon = 1e-100
    logPs_deep_abyss_maf = np.sort(-np.log10(Ps_deep_abyss_maf+epsilon))
    deep_abyss_maf_df = pd.DataFrame({'coeff':coefs_deep_abyss_maf, "AFs":AFs_deep_abyss_maf, "Ps_deep_abyss_maf": Ps_deep_abyss_maf})
    n = len(deep_abyss_maf_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    deep_abyss_maf_df['expected_P'] = expected_quantiles
    deep_abyss_maf_df['logPs_deep_abyss_maf'] = logPs_deep_abyss_maf
    deep_abyss_maf_df['expected_logP'] = expected_logP
    return deep_abyss_maf_df

def deep_abyss_pred_linreg(complete, y, deep_abyss_pred):
    y = y - np.array(deep_abyss_pred).flatten()
    Ps_deep_abyss_pred = []
    intercepts_deep_abyss_pred = []
    snps_deep_abyss_pred = []
    AFs_deep_abyss_pred = []
    coefs_deep_abyss_pred = []
    faulty_snps_deep_abyss_pred = []
    
    for snp in list(complete.columns):
        index_to_keep = snp.split("_AF_")[0]
        X_snp = np.array(list(complete[snp]))
        #X = np.column_stack((X_snp, np.array(deep_abyss_pred).flatten()))
        X = sm.add_constant(X_snp)
        try:
            #intercept, beta_hat, se_beta, t_values, p_values = lin_reg(pheno, geno)
            model = sm.OLS(y, X).fit()
            p_values = model.pvalues[1]
            betas = model.params[1:]
            beta_hat = betas[0]
            se_beta = model.bse[1]
            t_values = model.tvalues[1]
            Ps_deep_abyss_pred.append(p_values)
            snps_deep_abyss_pred.append(snp.split("_AF_")[0])
            AFs_deep_abyss_pred.append(snp.split("_AF_")[1])
            coefs_deep_abyss_pred.append(beta_hat)
        except Exception as e:
            faulty_snps_deep_abyss_pred.append(snp)
            print(e)

    Ps_deep_abyss_pred = np.sort(Ps_deep_abyss_pred)
    epsilon = 1e-100
    logPs_deep_abyss_pred = np.sort(-np.log10(Ps_deep_abyss_pred+epsilon))
    deep_abyss_pred_df = pd.DataFrame({'coeff':coefs_deep_abyss_pred, "AFs":AFs_deep_abyss_pred, "Ps_deep_abyss_pred": Ps_deep_abyss_pred})
    n = len(deep_abyss_pred_df)
    expected_quantiles = np.arange(1, n + 1) / n
    expected_logP = np.sort(-np.log10(expected_quantiles+epsilon))
    deep_abyss_pred_df['expected_P'] = expected_quantiles
    deep_abyss_pred_df['logPs_deep_abyss_pred'] = logPs_deep_abyss_pred
    deep_abyss_pred_df['expected_logP'] = expected_logP
    return deep_abyss_pred_df

