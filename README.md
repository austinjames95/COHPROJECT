# Instillation Instructions:
---
Before using the program simpleITK is required, please run the following command:
## pip install --target=lib simpleitk
---
After you run this command you must set the lib folder as the directory. To do this run the following command: 
##

$env:PYTHONPATH = "lib"
---
# Setup Instructions 
---
This program is designed to compare different timepoints of the same patient, in this program we just call these entirely different patients (i.e. Patient1 and Patient2). Simply export patient data to each respective Patient in the Input folder. Finally add your elastix file directory into the config file with the format r"example\format" 
