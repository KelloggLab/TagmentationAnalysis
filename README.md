# TagmentationAnalysis

This Jupyter notebook analyzes tagmentation sequencing data, performing alignment, quantification, and visualization of insertion patterns. It is designed to be modular and adaptable to different experimental setups.

## Features

- Aligns reads to reference sequences using `bbmap` and `samtools`
- Extracts and quantifies integration indices
- Plots strand-specific insertion site histograms
- Outputs plots and data summaries to a specified directory

## Getting Started

### Installation

Clone this repository and create a new conda environment:

conda create -n integration_env python=3.10 biopython matplotlib numpy bbmap samtools notebook jupyterlab -c conda-forge -c bioconda -y

## Launch the Notebook
Activate the environment and launch Jupyter:

jupyter notebook

If you encounter an architecture-related error, try:

PYTHONNOUSERSITE=1 conda run -n integration_env jupyter notebook

Then, navigate to the location of the notebook and open CAST_TagmentationNotebook.ipynb.

## Usage
1. Modify the input parameters (e.g., input FASTA/FASTQ files, output directory) in the first cells of the notebook.

2. Run each cell sequentially.

3. Output files including histograms and processed data will be saved in the specified output directory.

## Dependencies
This notebook requires the following packages (automatically installed with the conda environment above):

biopython
matplotlib
numpy
bbmap
samtools
notebook
jupyterlab
