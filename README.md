# Instillation Instructions:
---
The program was written in python 3.10.11, please run the following commands to run this program

## python3 -m venv venv
venv\Scripts\activate.bat

After the virtual eviroment is created and activated run the .bat file found in the repo, there will be libraries installed in both conda and pip as there are compatibility issues with some libraries on conda.
If the bat file does not work, just install all of the libraries that are listed inside of the bat file.

---
# Setup Instructions 
---
This program is designed to compare different timepoints of the same patient, in this program we just call these entirely different patients (i.e. Patient1 and Patient2). Simply export patient data to each respective Patient in the Input folder. Finally add your elastix file directory into the config file with the format r"example\format" 

