"""Merge all CSVs in the current folder into a single file."""

from glob import glob

# Append all CSVs (excluding the output file) into main.csv
# Writes the header once and skips duplicate headers in subsequent files.
output_file = 'main.csv'

csv_files = sorted(glob('*.csv'))
csv_files = [csv for csv in csv_files if csv != output_file]

with open(output_file, 'w') as singleFile:
    wrote_header = False
    for csv in csv_files:
        with open(csv, 'r') as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    if wrote_header:
                        continue
                    wrote_header = True
                singleFile.write(line)