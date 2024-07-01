# Import libraries
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import importlib.util
import sys
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import functions from helpers module
from helpers import parse_variables, get_risk_level, hi_gauss_blob_risk_fun, blob_risk_fun, NW_risk_fun, square_risk_fun, simulate_cc_status
args = sys.argv

# Parse arguments
if len(args) > 1:
    iterations = int(args[1])
else:
    pass
# Declare functions
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

dfs = []
complete = pd.read_pickle(f"../data/genotype/simulated_complete_genotypes_AF_0_0.5_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")
complete = complete/2

AF_prob = pd.read_pickle(f"../data/estimated p values/estimated_af_deep_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")


for i in range(iterations):    
    simulated_case_control = simulate_cc_status(geno, mu, effect, populations[name_risk])
    populations['case_control_env'] = simulated_case_control


    Ps_abyss_deep = []
    snps_abyss_deep = []

    # Run deep_Abyss









    
    for snp in list(complete.columns):
        geno = np.array(complete[snp])
        try:
            [intercept, coefficients, p_values] = log_reg(populations['case_control_env'], geno, AF_prob[snp])
            snps_abyss_deep.append(snp)
            Ps_abyss_deep.append(p_values[1])
        except Exception as e:
            print(e)
    

    df_abyss_deep = pd.DataFrame(data={'snps': snps_abyss_deep, 'P_vals_abyss_deep': Ps_abyss_deep})
    log10_p_abyss_deep = np.sort(-np.log10(df_abyss_deep['P_vals_abyss_deep']))
    df_abyss_deep['log10_p_abyss_deep_sorted'] = log10_p_abyss_deep
    dfs.append(df_abyss_deep)


df_abyss_deep = pd.DataFrame()
for i in range(len(dfs)):
    try:
        df_abyss_deep[f"{i}_abyss_deep"] = dfs[i]["log10_p_abyss_deep_sorted"]
        df_abyss_deep["snps"] = dfs[i]["snps"]
    except Exception as e:
        print(e)

log_columns = [col for col in df_abyss_deep.columns if '_abyss_deep' in col]
mean_log = df_abyss_deep[log_columns].mean(axis=1)
df_abyss_deep = df_abyss_deep[['snps']]
df_abyss_deep["-logP_abyss_deep"] = mean_log
df_abyss_deep.to_pickle(f"../data/iterative_pvalues/pvals_abyss_deep_pheno_{name_risk}_G{G}_L{L}_c{c}_k{k}_M{M}.pkl")