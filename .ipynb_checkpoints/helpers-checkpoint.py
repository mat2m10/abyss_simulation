import numpy as np

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

def simulate_quant_trait(mu, genotypes, beta=0, env=0):
    mean = mu + np.dot(genotypes,beta) + env

    true_mean = sum(mean) / len(mean)
    mean = mean - true_mean
    trait = []
    for element in mean:
        trait.append(np.random.normal(element,0.7))
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
    