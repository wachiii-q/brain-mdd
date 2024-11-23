# BRAIN-MDD
 Brain-MDD is a python-based classification of vulnerability to major depressive disorder (MDD) using electroencephalography (EEG). We examine EEG biomarkers through both 
 conventional machine-learning and deep-learning approachs. 

## Getting Started

### Prerequisites
- miniconda is required. You can download it [here](https://docs.conda.io/en/latest/miniconda.html).

### Installing (Only for first time)
1. Clone this repo

2. Get to project directoru

        cd ./brain-mdd

3. Create a new virtual environment using `conda`

        conda create --name brain-mdd python=3.12

4. Activate the virtual environment

        conda activate brain-mdd

5. install packages

        pip install -e .

### Environments Update

Sometimes, we add new packages to the project. So, you need to update your environment to get the latest packages. It's recommended to update your environment every time you pull the latest code.

    pip install -e .

### Reinstalling

When thing broken, you can try to reinstall the package. And please make sure you're in the right virtual environment, using right version of python.

    pip install -e . --force-reinstall
