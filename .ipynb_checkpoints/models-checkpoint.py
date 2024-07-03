import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import warnings
from scipy.stats import t
from scipy import stats
import statsmodels.api as sm
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Input, Model, layers, regularizers
warnings.filterwarnings("ignore")

def ols_regression(y, X1, covs=None):
    """
    Perform OLS regression with one primary predictor X1 and optional covariates.

    Parameters:
    y (pd.Series): The dependent variable.
    X1 (pd.Series): The primary predictor.
    covs (pd.DataFrame or None): DataFrame where each column is a covariate. Can be None.

    Returns:
    None: Prints the coefficients and p-values.
    """
    # Combine X1 and covariates into a single DataFrame
    if covs is not None:
        X = pd.concat([X1, covs], axis=1)
    else:
        X = X1.to_frame()
    
    X = sm.add_constant(X)  # Adds a column of ones to include an intercept in the model

    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract coefficients (beta values) and p-values
    beta_values = results.params
    p_values = results.pvalues
    
    return beta_values, p_values

def check_columns(geno, covs):
    # Check if covs is None
    if covs is None:
        print(f"No Covs!")
        return False
    
    # Check if covs is a DataFrame and its columns match geno's columns
    if isinstance(covs, pd.DataFrame):
        
        if list(geno.columns) == list(covs.columns):
            print(f"Abyss!")
            return True
        else:
            print(f"Covs")
            return False
    
    # Return False if covs is neither None nor a DataFrame
    print(f"Nothing works!")
    return False

def manhattan_linear(geno, y, covs=None):
    Ps = []
    snps = []
    coefs = []
    AFs = []
    # check if you have snp specific covariates

    if check_columns(geno, covs):
        # rename to not get the correct betas and p-values back
        covs_with_suffix = covs.add_suffix('_covariate')
        for snp in list(geno.columns):
            X = geno[snp]
            betas, p_values = ols_regression(y, X, covs_with_suffix[[f"{snp}_covariate"]])
            
            coefs.append(betas[snp])
            Ps.append(p_values[snp])
            
            snps.append(snp.split("_AF_")[0])
            AFs.append(snp.split("_AF_")[1])
    else:
        for snp in list(geno.columns):
            X = geno[snp]
            betas, p_values = ols_regression(y, X, covs)
            
            coefs.append(betas[snp])
            Ps.append(p_values[snp])
            
            snps.append(snp.split("_AF_")[0])
            AFs.append(snp.split("_AF_")[1])
    logPs = -np.log10(Ps)
    df = pd.DataFrame({'snp':snps,'coefs':coefs, "AFs":AFs, "Ps": Ps, "-logPs": logPs})
    return df

def gc(df):
    Ps = np.array(df['Ps'])
    coefs = list(df['coefs'])
    AFs = list(df['AFs'])
    median_chi2 = np.median(stats.chi2.ppf(1 - Ps, 1))
    expected_median = stats.chi2.ppf(0.5, 1)
    lambda_gc = median_chi2 / expected_median
    
    # Convert p-values to chi-square statistics
    chi2_values = stats.chi2.isf(Ps, df=1)
    
    # Adjust chi-square statistics
    chi2_corr = chi2_values / lambda_gc
    
    # Convert adjusted chi-square statistics back to p-values
    Ps = stats.chi2.sf(chi2_corr, df=1)
    logPs = -np.log10(Ps)
    GC_df = pd.DataFrame({'coefs':coefs, "AFs":AFs, "Ps": Ps, "-logPs": logPs})
    return GC_df

