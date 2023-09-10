import os 
import glob
import numpy as np


# grab the name of files in the directory 'chrom'
# excluding the file with 'html' extension
# and store the names in a list

# get the current working directory
cwd = os.getcwd()
chrom_files = glob.glob(cwd + '/Program-HW2/chrom/*')
print(chrom_files)
chrom_files = [x for x in chrom_files if 'html' not in x]
print(f'# of files: {len(chrom_files)}')

# files ending with a are training data
# files ending with b are testing data

# list of training data
train_files = [x for x in chrom_files if 'a' in x]
print(f'# of training files: {len(train_files)}')

# list of testing data
test_files = [x for x in chrom_files if 'b' in x]
print(f'# of testing files:  {len(test_files)}')

# print the first 5 files in the training data
print(f'First 5 files in the training data: {train_files[:5]}')
