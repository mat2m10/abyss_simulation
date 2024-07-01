# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
import importlib.util
import sys

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from helpers module
from helpers import parse_variables, get_risk_level, hi_gauss_blob_risk_fun, blob_risk_fun, NW_risk_fun, square_risk_fun, simulate_cc_status

# Parse variables
dict = parse_variables('../geno_simulation.txt')
G = int(dict['G'])
L = int(dict['L'])
c = int(dict['c'])
k = int(dict['k'])
M = float(dict['M'])


# Thresholds
very_rare_threshold_L = float(dict['very_rare_threshold_L'])
very_rare_threshold_H = float(dict['very_rare_threshold_H'])

rare_threshold_L = float(dict['rare_threshold_L'])
rare_threshold_H = float(dict['rare_threshold_H'])

common_threshold_L = float(dict['common_threshold_L'])
common_threshold_H = float(dict['common_threshold_H'])

populations = pd.read_pickle(f"../data/phenotype/simulated_population_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
path_risk="../pheno_simulation.txt"
risk_level = get_risk_level(path_risk)
risk_level = risk_level.split("\n")[-1]
print(risk_level)
# Define the module name and file path
module_name = 'helpers'
module_file_path = '../helpers.py'  # Replace '../path/to/helpers.py' with the actual path to helpers.py

# Load the module dynamically
module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
helpers = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helpers)
# Get the function dynamically
risk_function = getattr(helpers, risk_level)

name_risk = risk_level.split('_fun')[0]
populations['x_temp'] = populations['x']/k
populations['y_temp'] = populations['y']/k
populations[name_risk] = list(populations.apply(lambda row: risk_function(row['x_temp'], row['y_temp']), axis=1))
populations[name_risk] = populations[name_risk].astype('float')

geno = 0
mu = 0
effect = 0




for i in range(4):
    simulated_case_control = simulate_cc_status(geno, mu, effect, populations[name_risk])
    populations['case_control_env'] = simulated_case_control
    populations[[name_risk, 'case_control_env']].to_pickle(f"../data/phenotype/simulatedcase_control_onlyenvrisk_{name_risk}_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    
    
    complete = pd.read_pickle(f"../data/genotype/simulated_complete_genotypes_AF_0_0.5_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    very_rare= pd.read_pickle(f"../data/genotype/simulated_veryrare_genotype_AF_{very_rare_threshold_L}_{very_rare_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    rare = pd.read_pickle(f"../data/genotype/simulated_rare_genotype_AF_{rare_threshold_L}_{rare_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    common = pd.read_pickle(f"../data/genotype/simulated_common_genotype_AF_{common_threshold_L}_{common_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    
    complete = complete/2
    
    path_risk="../pheno_simulation.txt"
    risk_level = get_risk_level(path_risk)
    risk_level = risk_level.split("\n")[-1]
    name_risk = risk_level.split('_fun')[0]
    
    risk = pd.read_pickle(f"../data/phenotype/simulatedcase_control_onlyenvrisk_{name_risk}_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    
    def log_reg(y, x, covs=None):
        # Suppress optimization termination message
        warnings.filterwarnings("ignore")
        # Fit the logistic regression model
        model = LogisticRegression()
        if covs is None or len(covs) == 0:
            model.fit(x.reshape(-1, 1), y)
            X = x.reshape(-1, 1)
        else:
            X = np.column_stack((x, covs))
            model.fit(X, y)
        
        # Extract coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Statsmodels logistic regression
        X2 = sm.add_constant(X)
        logit_model = sm.Logit(y, X2)
    
        # Suppress optimization termination message
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logit_result = logit_model.fit(disp=0)
        
        p_values = logit_result.pvalues
        betas = logit_result.params
        
        # Return the results without printing
        return [intercept, coefficients, p_values]


    # No correction, no risk
    
    # No correction
    
    coefs = []
    intercepts = []
    faulty_snps_no_corr = []
    Ps_no_corr = []
    snps_no_corr = []

    for snp in list(complete.columns):
        geno = np.array(complete[snp])
        try:
            [intercept, coefficients, p_values] = log_reg(risk['case_control_env'], geno)
            snps_no_corr.append(snp)
            coefs.append(coefficients)
            intercepts.append(intercept)
            Ps_no_corr.append(p_values[1])
        except Exception as e:
            print(e)
            faulty_snps_no_corr.append(snp)
    
    PC_complete = pd.read_pickle(f"../data/phenotype/PC_and_pop_simulated_complete_genotypes_AF_0_0.5_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    PC_veryrare = pd.read_pickle(f"../data/phenotype/PC_and_pop_simulated_veryrare_genotype_AF_{very_rare_threshold_L}_{very_rare_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    PC_rare = pd.read_pickle(f"../data/phenotype/PC_and_pop_simulated_rare_genotype_AF_{rare_threshold_L}_{rare_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
    PC_common = pd.read_pickle(f"../data/phenotype/PC_and_pop_simulated_rare_genotype_AF_{common_threshold_L}_{common_threshold_H}_simulated_geno_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")

    Ps_PCs = []
    snps_PCs = []
    coefs = []
    intercepts = []
    faulty_snps_PCs = []
    for snp in list(complete.columns):
        geno = np.array(complete[snp])
        try:
            [intercept, coefficients, p_values] = log_reg(risk['case_control_env'], geno, PC_complete[['PC1','PC2','PC3','PC4','PC5']])
            snps_PCs.append(snp)
            coefs.append(coefficients)
            intercepts.append(intercept)
            Ps_PCs.append(p_values[1])
        except Exception as e:
            print(e)
            faulty_snps_PCs.append(snp)


    Ps_rare_PCs = []
    snps_rare_PCs = []
    coefs = []
    intercepts = []
    faulty_snps_rare_Pcs = []
    for snp in list(complete.columns):
        geno = np.array(complete[snp])
        try:
            [intercept, coefficients, p_values] = log_reg(risk['case_control_env'], geno, PC_veryrare[['PC1','PC2','PC3','PC4','PC5']])
            snps_rare_PCs.append(snp)
            coefs.append(coefficients)
            intercepts.append(intercept)
            Ps_rare_PCs.append(p_values[1])
        except Exception as e:
            print(e)
            faulty_snps_rare_Pcs.append(snp)

    df_Ps_no_corr = pd.DataFrame(data={'snps': snps_no_corr, 'P_vals_nocorr': Ps_no_corr})
    df_Ps_PCs = pd.DataFrame(data={'snps': snps_PCs, 'P_vals_PCs': Ps_PCs})
    df_Ps_rare_PCs = pd.DataFrame(data={'snps': snps_rare_PCs, 'P_vals_rare_PCs': Ps_rare_PCs})

    # Merge df and df_PC on 'snps'
    merged_df = pd.merge(df_Ps_no_corr, df_Ps_PCs, on='snps')
    
    # Merge df_Ps_PCs with the already merged dataframe on 'snps'
    final_merged_df = pd.merge(merged_df, df_Ps_rare_PCs, on='snps')

    log10_p_no_corr = np.sort(-np.log10(final_merged_df['P_vals_nocorr']))
    log10_p_PCs = np.sort(-np.log10(final_merged_df['P_vals_PCs']))
    log10_p_rare_PCs = np.sort(-np.log10(final_merged_df['P_vals_rare_PCs']))

    expected_log10_p = np.sort(-np.log10((np.arange(len(log10_p_no_corr)) + 1) / (len(log10_p_no_corr) + 1)))
    final_merged_df['log10_p_no_corr_sorted'] = log10_p_no_corr
    final_merged_df['log10_p_PCs_sorted'] = log10_p_PCs
    final_merged_df['log10_p_rare_PCs_sorted'] = log10_p_rare_PCs
    final_merged_df['log10_p_expected_sorted'] = expected_log10_p

    os.makedirs(f"../data/iterative_pvalues", exist_ok=True)
    temp = list(final_merged_df["P_vals_nocorr"])
    expected_p = np.sort((np.arange(len(temp)) + 1) / (len(temp) + 1))
    final_merged_df["P_vals_expected"] = expected_p
    final_merged_df["P_vals_nocorr_sorted"] = np.sort(final_merged_df["P_vals_nocorr"])
    final_merged_df["P_vals_PCs_sorted"] = np.sort(final_merged_df["P_vals_PCs"])
    final_merged_df["P_vals_rare_PCs_sorted"] = np.sort(final_merged_df["P_vals_rare_PCs"])
    final_merged_df.to_pickle(f"../data/iterative_pvalues/{i}_pvals_nocorr_PC_rarePC_pheno_{name_risk}_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")