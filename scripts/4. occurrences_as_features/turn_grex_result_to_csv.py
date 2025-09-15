import json
import csv

# File paths
input_file = '/Users/madalina/Documents/M2TAL/stage/grex/ro_verbs_cluster1.json'
output_file = '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/output/grex_predictors_per_cluster/ro_verbs_cluster1.csv'

# Read JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract rules
rules = data['rules']

# Define CSV headers
headers = [
    "pattern", "n_pattern_occurence", "n_pattern_positive_occurence", "decision", 
    "alpha", "value", "coverage", "precision", "delta", "g-statistic", "p-value", "cramers_phi"
]

# Write to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for rule in rules:
        writer.writerow(rule)

print(f"Data has been written to {output_file}")