import numpy as np
import pandas as pd
import subprocess
import os

# Function to pair SNPs and summarize genotype
def summarize_genotypes(df):
    summarized_genotypes = {}
    # Iterate over pairs of columns
    for i in range(1, df.shape[1], 2):
        pair_sum = df.iloc[:, i-1] + df.iloc[:, i]
        # Apply the genotype summarization logic
        summarized_genotypes[f'G{i//2 + 1}'] = np.where(pair_sum == 2, 2, pair_sum)
    return pd.DataFrame(summarized_genotypes)

# Function to flip 0s to 2s and 2s to 0s
def flip_genotypes(row):
    if row['AFs'] > 0.5:
        # Apply transformation for the condition
        row[:-1] = row[:-1].replace({0: 2, 2: 0})
        row['AFs'] = 1 - row['AFs']  # Adjust allele frequency
    return row

def contains_all_genotypes(series, genotypes={0.0, 1.0, 2.0}):
    return genotypes.issubset(series.unique())
    
def simulate_genos(G, L, c, k, M, HWE):
    dict = parse_variables('geno_simulation.txt')
        # Define the R commands to run, passing parameters as arguments
    commands = [
        f"source('geno_simulation.txt')",
        f"source('create_geno.R', echo=TRUE)",
    ]
    
    
    commands = [
        "source('geno_simulation.txt')",
        f"G <- {G}",
        f"L <- {L}",
        f"c <- {c}",
        f"k <- {k}",
        f"M <- {M}",
        "source('create_geno.R', echo=TRUE)"
    ]
    
    # Concatenate commands into a single string
    r_script = ";".join(commands)
    
    # Run the R script
    result = subprocess.run(['Rscript', '-e', r_script], capture_output=True, text=True)
    
    # Print the output
    #print(result.stdout)
    
    # Check for errors
    if result.returncode != 0:
        print("Error executing R script:")
        print(result.stderr)
        pass
    
    os.makedirs(f"data/concept/genotype/raw",exist_ok=True)
    os.system(f"mv simulated_genotypes_G{G}_L{L}_c{c}_k{k}_M{M}.csv data/concept/genotype/raw/")
    
    # Thresholds
    very_rare_threshold_L = float(dict['very_rare_threshold_L'])
    very_rare_threshold_H = float(dict['very_rare_threshold_H'])
    
    rare_threshold_L = float(dict['rare_threshold_L'])
    rare_threshold_H = float(dict['rare_threshold_H'])
    
    common_threshold_L = float(dict['common_threshold_L'])
    common_threshold_H = float(dict['common_threshold_H'])
    
    file = f"data/concept/genotype/raw/simulated_genotypes_G{G}_L{L}_c{c}_k{k}_M{M}.csv"
    path_simulated_file = "./"+ file

    
    number_of_loci = G*L
    number_of_individuals = c*k*k
    simulated_loci= pd.read_csv(path_simulated_file)


    # Apply the function to the sample DataFrame
    simulated_genotype = summarize_genotypes(simulated_loci)
    columns_to_drop  = simulated_genotype.columns[simulated_genotype.nunique() == 1] # If double columns delete it 
    simulated_genotype = simulated_genotype.drop(columns=columns_to_drop)
    
    number_of_populations = k*k
    labels_pop = []
    for i in range(number_of_populations):
        labels_pop += [i+1]*c
    
    simulated_genotype["populations"] = labels_pop

    # Uncomment if you want mixed populations
    #simulated_genotype['populations'] = simulated_genotype['populations'].apply(lambda x: 1 if x in [1, 2] else 2)
    
    unique_pops = simulated_genotype['populations'].unique()
    unique_pops.sort()
    dfs = []
    required_values = {0, 1, 2}
    
    # Optimization: Cache the set operation result
    simulated_genotype_sets = {col: set(simulated_genotype[col]) for col in simulated_genotype.columns}

    if HWE == 1:
        for pop in unique_pops:
            temp_pop = simulated_genotype[simulated_genotype["populations"] == pop].drop('populations', axis=1)
            
            for col in temp_pop.columns:
                column_values = simulated_genotype_sets[col]
                
                if not required_values.issubset(column_values):
                    # Optimization: Vectorized random choice and assignment
                    indices = np.random.choice(temp_pop.index, size=3, replace=False)
                    temp_pop.loc[indices[0], col] = 0
                    temp_pop.loc[indices[1], col] = 1
                    temp_pop.loc[indices[2], col] = 2
    
                # Calculate frequencies
                value_counts = temp_pop[col].value_counts().reindex([0, 1, 2], fill_value=0)
                total = value_counts.sum()
                q = (2*value_counts[2] + value_counts[1])/ (2*total)
                if q > 0.5:
                    q = 1-q
                p = 1 - q
                freq_maj = p ** 2
                freq_het = 2 * p * q
                freq_min = q ** 2
    
                genotypes = [-1.0, 0.0, 1.0]  # List of all possible genotypes
                
                # Optimization: Vectorized assignment of new genotypes
                pop_geno = np.random.choice(genotypes, size=total, p=[freq_maj, freq_het, freq_min])

                # Check for missing genotypes and replace elements until all genotypes are present
                missing_genotypes = [g for g in genotypes if g not in pop_geno]
                
                # Use a while loop to ensure no genotype is missing
                while missing_genotypes:
                    for missing in missing_genotypes:
                        # Randomly choose an index to replace
                        idx = np.random.randint(0, total)
                        pop_geno[idx] = missing
                    
                    # Recheck which genotypes are missing after replacement
                    missing_genotypes = [g for g in genotypes if g not in pop_geno]
                
                temp_pop[col] = pop_geno
    
            dfs.append(temp_pop)
    
    else:
        print("HWE")
        for pop in unique_pops:
            temp_pop = simulated_genotype[simulated_genotype["populations"] == pop].drop('populations', axis=1)
            
            for col in temp_pop.columns:
                column_values = simulated_genotype_sets[col]
                
                if not required_values.issubset(column_values):
                    # Optimization: Vectorized random choice and assignment
                    indices = np.random.choice(temp_pop.index, size=3, replace=False)
                    temp_pop.loc[indices[0], col] = 0
                    temp_pop.loc[indices[1], col] = 1
                    temp_pop.loc[indices[2], col] = 2
    
                # Calculate frequencies
                value_counts = temp_pop[col].value_counts().reindex([0, 1, 2], fill_value=0)
                total = value_counts.sum()
                q = (2*value_counts[2] + value_counts[1])/ (2*total)
                if q > 0.5:
                    q = 1-q
                p = 1 - q
                freq_maj = p ** 2
                freq_het = q ** 2
                freq_min = 2 * p * q

                genotypes = [-1.0, 0.0, 1.0]  # List of all possible genotypes
                
                # Optimization: Vectorized assignment of new genotypes
                pop_geno = np.random.choice(genotypes, size=total, p=[freq_maj, freq_het, freq_min])

                # Check for missing genotypes and replace elements until all genotypes are present
                missing_genotypes = [g for g in genotypes if g not in pop_geno]
                
                ## Use a while loop to ensure no genotype is missing
                #while missing_genotypes:
                #    for missing in missing_genotypes:
                #        # Randomly choose an index to replace
                #        idx = np.random.randint(0, total)
                #        pop_geno[idx] = missing
                    
                    # Recheck which genotypes are missing after replacement
                #    missing_genotypes = [g for g in genotypes if g not in pop_geno]
                
                temp_pop[col] = pop_geno
    
            dfs.append(temp_pop)
    
    # Concatenate all dataframes if needed
    simulated_genotype = pd.concat(dfs, ignore_index=True)
    simulated_genotype = simulated_genotype+1
    
    # calculate when AF is > 0.5 and change the genotype
    # Initialize a dictionary to store allele frequencies
    allele_frequencies = {}
    
    # Calculate allele frequencies for each SNP column
    for snp in simulated_genotype.columns:
        total_alleles = 2 * len(simulated_genotype[snp])  # Total number of alleles (2 alleles per sample)
        minor_allele_count = (2 * simulated_genotype[snp].value_counts().get(0, 0)) + simulated_genotype[snp].value_counts().get(1, 0)
        allele_frequency = minor_allele_count / total_alleles
        allele_frequencies[snp] = allele_frequency
    
    temp = simulated_genotype.T
    temp['AFs'] = allele_frequencies
    
    # Apply the function across the DataFrame, row-wise
    df_transformed = temp.apply(flip_genotypes, axis=1)
    
    simulated_genotype = df_transformed.drop('AFs', axis=1).T
    columns_to_drop  = simulated_genotype.columns[simulated_genotype.nunique() == 1] # If double columns delete it 
    simulated_genotype = simulated_genotype.drop(columns=columns_to_drop)
    
    simulated_genotype = simulated_genotype[[col for col in simulated_genotype.columns if contains_all_genotypes(simulated_genotype[col])]]
    
    # calculate when AF is > 0.5 and change the genotype
    # Initialize a dictionary to store allele frequencies
    allele_frequencies = {}
    
    # Calculate allele frequencies for each SNP column
    for snp in simulated_genotype.columns:
        total_alleles = 2 * len(simulated_genotype[snp])  # Total number of alleles (2 alleles per sample)
        minor_allele_count = (2 * simulated_genotype[snp].value_counts().get(0, 0)) + simulated_genotype[snp].value_counts().get(1, 0)
        allele_frequency = minor_allele_count / total_alleles
        allele_frequencies[snp] = allele_frequency
    
    temp = simulated_genotype.T
    temp['AFs'] = allele_frequencies
    AFs = temp[['AFs']]
    
    # Create slices as copies to avoid SettingWithCopyWarning
    very_rare = temp[(temp['AFs'] > very_rare_threshold_L) & (temp['AFs'] <= very_rare_threshold_H)].copy()
    rare = temp[(temp['AFs'] > rare_threshold_L) & (temp['AFs'] <= rare_threshold_H)].copy()
    common = temp[(temp['AFs'] > common_threshold_L) & (temp['AFs'] <= common_threshold_H)].copy()
    
    # Modify 'snps' column using .loc to avoid warnings
    very_rare.loc[:, 'snps'] = very_rare.index + '_AF_' + very_rare['AFs'].astype(str)
    very_rare.set_index('snps', inplace=True)
    very_rare_to_save = very_rare.drop('AFs', axis=1).T
    very_rare_afs = very_rare[['AFs']]
    
    rare.loc[:, 'snps'] = rare.index + '_AF_' + rare['AFs'].astype(str)
    rare.set_index('snps', inplace=True)
    rare_to_save = rare.drop('AFs', axis=1).T
    rare_afs = rare[['AFs']]
    
    common.loc[:, 'snps'] = common.index + '_AF_' + common['AFs'].astype(str)
    common.set_index('snps', inplace=True)
    common_to_save = common.drop('AFs', axis=1).T
    common_afs = common[['AFs']]
    
    very_rare_to_save = very_rare_to_save.rename(columns=lambda x: 'VR' + x)/2
    rare_to_save = rare_to_save.rename(columns=lambda x: 'R' + x)/2
    common_to_save = common_to_save.rename(columns=lambda x: 'C' + x)/2
    complete = pd.concat([common_to_save, rare_to_save, very_rare_to_save], axis=1)
    complete = ((complete*2)-1)


    return very_rare_to_save, rare_to_save, common_to_save, complete




def calculate_true_maf_per_pop(genos, humans):
    geno = genos.copy()
    geno['pop'] = humans['populations']

    p2s_dfs = []
    q2s_dfs = []
    twopqs_dfs = []
    
    for pop in geno['pop'].unique():
        temp = geno[geno['pop'] == pop].drop("pop", axis=1)
    
        # Count the number of major, heterozygous, and minor alleles
        counts = temp.apply(pd.Series.value_counts).fillna(0)
    
        num_maj = counts.loc[1.0]
        num_het = counts.loc[0.0]
        num_min = counts.loc[-1.0]
    
        total_humans = num_maj + num_het + num_min
    
        # Normalize to get frequencies instead of counts
        p2s = num_maj / total_humans
        twopqs = num_het / total_humans
        q2s = num_min / total_humans
    
        # Expand the normalized values across all rows for each population
        p2s_dfs.append(pd.DataFrame([p2s] * temp.shape[0], index=temp.index, columns=temp.columns))
        twopqs_dfs.append(pd.DataFrame([twopqs] * temp.shape[0], index=temp.index, columns=temp.columns))
        q2s_dfs.append(pd.DataFrame([q2s] * temp.shape[0], index=temp.index, columns=temp.columns))
        
    # Drop "pop" from the original DataFrame
    geno = geno.drop("pop", axis=1)
    
    # Concatenate all population-specific DataFrames
    true_p2s = pd.concat(p2s_dfs)
    true_twopqs = pd.concat(twopqs_dfs)
    true_q2s = pd.concat(q2s_dfs)
    return true_p2s, true_twopqs, true_q2s


# Function to parse variables from the text file
def parse_variables(file_path):
    variables = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.strip() == '' or line.startswith('#'):
                continue
            # Split the line by '<-' to get variable name and value
            name, value = line.split('<-')
            # Remove leading and trailing whitespace from name and value
            name = name.strip()
            value = value.strip()
            # Convert value to appropriate data type
            try:
                value = int(value)  # Try converting to integer
            except ValueError:
                try:
                    value = float(value)  # Try converting to float
                except ValueError:
                    pass  # If conversion fails, keep it as string
            # Store variable in dictionary
            variables[name] = value
    return variables



# Function to read the content of pheno_simulation.txt and determine the risk level
def get_risk_level(path_risk="pheno_simulation.txt"):
    with open(f"{path_risk}", 'r') as file:
        content = file.read().strip()
        return content


# Define a function to map values to colors
def map_to_color(x, y, z, df):
    # Example mapping logic: you can customize this based on your data and preferences
    r = x / df['x'].max()  # Red component based on 'x'
    g = y / df['y'].max()  # Green component based on 'y'
    b = 0.5  # Blue component based on 'z'
    return (r, g, b)

def simulate_quant_trait(mu, genotypes, beta=0, env=0, precision=0.1):
    mean = mu + np.dot(genotypes,beta) + env
    trait = []
    for element in mean:
        trait.append(np.random.normal(element,precision))
    return trait
    

def simulate_cc_status(geno, mu, beta_vector=0, env=0, snpXsnp=0):
    """
    Simulate case-control status for samples, given haploid genotypes,
    genotype risk (beta), environmental risk (env), and GE interaction term (interact),
    all in log odds units.
    
    Parameters:
    geno : numpy.ndarray
        Array containing the genotypes.
    mu : float
        Baseline log odds.
    beta_vector : numpy.ndarray, optional
        Genotype risk factor. Default is 0.
    env : numpy.ndarray, optional
        Environmental risk factor. Default is 0.
    snpXsnp : numpy.ndarray, optional
        SNP-SNP interaction vector. Default is 0.
        
    Returns:
    numpy.ndarray
        Array of simulated case-control status (0 or 1) for each sample.
    """
    # Calculate the log odds for each sample
    log_odds = mu + np.dot(geno, beta_vector) + env + snpXsnp
    # Stabilize the computation by subtracting the maximum value
    
    #max_log_odds = np.max(log_odds)
    #exp_term = np.exp(log_odds - max_log_odds)

    # Calculate probabilities using the stabilized exponential function
    probability = 1 / (1 + np.exp(-log_odds))

    # Simulate case-control status based on calculated probabilities
    status = np.random.binomial(1, probability)
    
    return status

"""
All the risk functions
"""
def no_risk_fun(x, y):
    return 0 * x

def NW_risk_fun(x, y):
    return 0.25 * (y - x + 1)

def N_risk_fun(x, y):
    return 0.5 * y

def blob_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.25)**2 + (y - 0.75)**2) < 0.25, 0.5, 0)

def center_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.5)**2 + (y - 0.5)**2) < 0.25, 0.5, 0)

def big_square_risk_fun(x, y):
    return np.where((y > 0.1) & (y < 0.5) & (x > 0.5) & (x < 0.9), 0.4, 0)

def square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76), 1, 0)

def hi_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76), 2, 0)

def mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.3) & (x > 0.6) & (x < 0.7), 1, np.where((y > 0.15) & (y < 0.35) & (x > 0.54) & (x < 0.76), 0.5, 0))

def mid_mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 0.7, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 0.35, 0)))

def mid_mid_mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 0.8, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 0.6, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 0.4, np.where((y > 0.04) & (y < 0.41) & (x > 0.44) & (x < 0.81), 0.2, 0))))

def big_bad_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.6) & (x > 0.35) & (x < 0.76), 1, 0)

# Re-definition of big_square_risk_fun with new logic
def big_square_risk_fun_updated(x, y):
    return 0.75 * np.where((y > 0.2) & (y < 0.3) & (x > 0.6) & (x < 0.7), 1, np.where((y > 0.15) & (y < 0.35) & (x > 0.54) & (x < 0.76), 1, 0))

def big_big_square_risk_fun(x, y):
    return 0.6 * np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 1, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 1, 0)))

def big_big_big_square_risk_fun(x, y):
    return 0.4285 * np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 1, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 1, np.where((y > 0.04) & (y < 0.41) & (x > 0.44) & (x < 0.81), 1, 0))))

def two_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76)) | ((y > 0.7) & (y < 0.86) & (x > 0.1) & (x < 0.26)), 1, 0)

def three_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76)) | ((y > 0.7) & (y < 0.86) & (x > 0.1) & (x < 0.26)) | ((y > 0.2) & (y < 0.36) & (x > 0.1) & (x < 0.26)), 1, 0)

def four_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.65) & (x < 0.8)) | ((y > 0.7) & (y < 0.86) & (x > 0.2) & (x < 0.36)) | ((y > 0.2) & (y < 0.36) & (x > 0.2) & (x < 0.36)) | ((y > 0.7) & (y < 0.86) & (x > 0.65) & (x < 0.8)), 1, 0)

def as_big_blob_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.25)**2 + (y - 0.75)**2) < 0.5, 0.5, 0)

def six_square_risk_fun(x, y):
    return np.where((y > 0.5) & (x < 0.7), 0.5, 0)

def gauss_blob_risk_fun(x, y):
    return 1 * np.exp(-((x - 0.25)**2 + (y - 0.75)**2) / (2 * 0.25**2))

def hi_gauss_blob_risk_fun(x, y):
    return 2 * np.exp(-((x - 0.25)**2 + (y - 0.75)**2) / (2 * 0.25**2))

def hi_hyperbole_risk_fun(x, y):
    return 2 *((x-0.5)**2 + (y-0.5)**2)

def hi_tangeant_risk_fun(x, y):
    return (x-0.5)**3 + (y-0.5)**3

def sine_risk_fun(x, y):
    f = 2
    return np.sin(f * (x+y)*2 * np.pi)