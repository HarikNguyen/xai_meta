import csv
import os
import scipy.stats as stats
import numpy as np

def log_to_csv(csv_path, log, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # write header if file does not exist and header is provided
        if not file_exists and header is not None:
            writer.writerow(header)

        # write log values
        writer.writerow(log)

def compute_stats(data):
    """Compute mean, std, ci95 for a list of numbers."""
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # sample standard deviation
    se = std / np.sqrt(n) # standard error
    ci95 = stats.t.ppf(0.975, n-1) * se # ci95 using t-distribution
    return mean, std, ci95
