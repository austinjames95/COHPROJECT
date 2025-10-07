# Instillation Instructions:
---
The program was written in python 3.10.18, this version is not avaliable on [python.com/downloads](https://www.python.org/downloads/), to install this version you must create a virtual enviroment (preferably in conda) with the following command

## conda create -n venv python=3.10.18
## conda activate venv

After the virtual eviroment is created and activated run the .bat file found in the repo, there will be libraries installed in both conda and pip as there are compatibility issues with some libraries on conda.

---
# Setup Instructions 
---
This program is designed to compare different timepoints of the same patient, in this program we just call these entirely different patients (i.e. Patient1 and Patient2). Simply export patient data to each respective Patient in the Input folder. Finally add your elastix file directory into the config file with the format r"example\format" 

