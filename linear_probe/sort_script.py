#!/usr/bin/env python3
"""Summarise linear-probe results into one table.

    python linear_probe/sort_script.py [logs/linear_probe]

Reads every ``*_results.csv`` written by linear_eval.py under the given
directory. The run label is the containing folder name (``<model>-<dataset>``);
the dataset comes from the CSV itself, so model names containing '-' are safe.
"""
import csv
import os
import sys

from prettytable import PrettyTable

base_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs/linear_probe'

METRICS = ['W_F1', 'AUROC', 'BACC', 'ACC', 'AUPR']
FIELDNAMES = ['Run', 'Dataset'] + METRICS

summary_data = []
for dirpath, _dirnames, filenames in os.walk(base_dir):
    for filename in sorted(filenames):
        if not filename.endswith('_results.csv'):
            continue
        run = os.path.relpath(dirpath, base_dir).replace(os.sep, '/')
        with open(os.path.join(dirpath, filename)) as csvfile:
            for row in csv.DictReader(csvfile):
                entry = {'Run': run, 'Dataset': row.get('Dataset', '')}
                for metric in METRICS:
                    value = row.get(metric, '')
                    try:
                        value = f'{float(value):.4f}'
                    except (TypeError, ValueError):
                        pass
                    entry[metric] = value
                summary_data.append(entry)

if not summary_data:
    sys.exit(f'No *_results.csv found under {base_dir}')

summary_data.sort(key=lambda x: (x['Run'], x['Dataset']))

table = PrettyTable()
table.field_names = FIELDNAMES
table.align = 'l'
for entry in summary_data:
    table.add_row([entry.get(field, '') for field in FIELDNAMES])

print(table)

output_txt = os.path.join(base_dir, 'summary_results.txt')
with open(output_txt, 'w') as txtfile:
    txtfile.write(str(table))
print(f'Summary results written to {output_txt}')
