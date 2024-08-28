# GMFOLD: Graph matching for DNA-aptamer secondary structure classification and machine learning interpretabiity

This repository contains the code and experiments to replicate the results presented in the paper titled "GMFOLD: Graph matching for DNA-aptamer secondary structure classification and machine learning interpretabiity"

**Anne Andrews**^[3], **Andrea L. Bertozzi**^[1, 2], **Paolo Climaco**^[4], **Noelle Mitchell**^[3], **Cameron Movassaghi**^[3] and **Matthew Tyler**^[2].

[1] California NanoSystems Institute, University of California, Los Angeles, 90095, CA, USA\
[2] Department of Mathematics, University of California, Los Angeles, 90095, CA, USA\
[3] Department of Chemistry and BioChemistry, Los Angeles, 90095, CA, USA\
[4] Institut für Numerische Simulation, Universität Bonn, Germany

Contact climaco@ins.uni-bonn for questions about code and data.

## Directory Structure
```plaintext

├── data/                           # Folder containing experimental data analyzed in our experiments
|    ├── dna.csv                        # CSV file with DNA sequences data from Seqfold github repository
|    ├── fold_published.csv             # CSV file with sequences and associated secondary structures and energies computed with GMfold
|    ├── Raw_files/                     # Subfolder containing raw, unprocessed experimental data
|    ├── published_clean_files/         # Subfolder with cleaned and processed data ready for analysis
|
├── notebooks/                      # Folder containing Jupyter notebooks for large scale data analysis 
|    ├── comparison.ipynb               # Notebook comparing different DNA folding algorithms: MGfold, Seqfold 2_0 and Seqfold
|    ├── folding_constraint.txt         # Text file outlining constraints required to get mfold/UNAfold energies
|    ├── fold_data.ipynb                # Notebook for processing and folding DNA sequences at scale
|    ├── read_excels_files.ipynb        # Notebook for reading and processing Excel files with experimental data
|    ├── Seq_vs_Seq2.ipynb              # Notebook comparing speed of Seqfold and  Seqfold 2.0 for DNA folding
|    ├── similarity_search.ipynb        # Notebook for similarity search of DNA secondary structures
|    ├── topic_modeling.ipynb           # Notebook applying topic modeling techniques 
|
├── src/                           # Folder containing source code for DNA folding
|    ├── dna.py                         # Script containing DNA enthalpy and entropy parameters foe energy computations
|    ├── gm_energy_functions.py         # Script containing energy functions used in MGfold
|    ├── GMfold.py                      # Main script for DNA folding using graph matching methods: Implementation of MGfold
|    ├── graph_matching.py              # Script for performing graph matching on DNA structures
|    ├── seqfold2_0.py                  # Script implementing the SeqFold 2.0 algorithm for DNA sequence folding
|    ├── Types.py                       # Script defining custom data types for DNA folding operations
|    ├── utils.py                       # Utility script with helper functions used across DNA folding workflows

└── environment.txt            # python packages required to run code.
└── README.md                   # Project README file.

```