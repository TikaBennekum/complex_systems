# complex_systems
Complex systems: River flooding dynamics

# How to install
## Python dependencies:
view pyproject.toml

## C++ compiling:
By far easiest on linux:
    
    apt install build-essential
    cd src/cpp_modules
    make


# Running experiments:
Python files are in the src directory
To run the "Lane's relation" experiments:
    experiments.py

To run the divergence/perturbation experiment:
    divergence_exp.py
    divergence_analysis.py


To run the number of streams / entropy experiments:
    experiments_num_streams.py

To run the (unsuccessful) SOC / avalanche experiments:
    experiments_abs_change.py