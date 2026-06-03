import csv
import os

def log_to_csv(csv_path, log, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # write header if file does not exist and header is provided
        if not file_exists and header is not None:
            writer.writerow(header)

        # write log values
        writer.writerow(log)
