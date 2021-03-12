# FUTON_Research
This is the home of my Master's thesis research project: FUTON, a proposed model for using Q-Learning to predict mortality in Sepsis patients.

## Build Instructions

The data for this project relies on several other research databases and projects to work.

1. Access to the database MIMIC-III is required to perform the Machine Learning (ML) techniques in this repo. Information on accessing MIMIC-III can be found here https://mimic.physionet.org/gettingstarted/access/.
    * After Access is granted, the MIMIC-III table must be created and loaded in PostgreSQL, scripts are provided here: https://github.com/MIT-LCP/mimic-     code/tree/master/buildmimic/postgres
2. These scripts are run through Juptyer Notebook using the Anaconda distribution + launcher. This can be found here: https://www.anaconda.com/products/individualhttps://www.anaconda.com/products/individual
    * Several packages required to run these scripts require some popular packages (NumPy, Pandas, MatPlotLib, Shelve) and some uncommon libraries (pymdptoolbox), for those not 
  installed on your local version, the Anaconda Prompt provided by the Anaconda distribution can be used to install anything missing using **pip install nameofpackage**
3. As the whole point of having the MIMIC-III database behind a certificate-locked access wall is to avoid public access, the Komoroski scripts located here: https://github.com/matthieukomorowski/AI_Clinician can be run in order to create an Excel file with the desired patient data and format used here. **Be Warned: This process can take several hours due to the amount of data being parsed and the languages being used**
    1. Download the full repo from github
    1. Run everything in AIClinician_Data_extract_MIMIC3_140219.ipynb, installing any missing packages using the above stated methods
    1. Run AIClinician_sepsis3_def_160219.m
    1. Run AIClinician_MIMIC3_dataset_160219.m, but append a line to the very end of the file reading `writetable(MIMICtable,'patient_data.csv','Delimiter',',');`
    1. This will extract the necessary file *patient_data.csv* used for the experiments here
4. The remainder of this repo is Juptyer Scripts and excess files used to run those materials.

## Contents of the Repo
A brief description of the various items available in this respository
### Add Sepsis Flags.ipynb
This generates a CSV file from *patient_data.csv* that appends 3 columns to the original. One for determining qSOFA score (https://qsofa.org/what.phphttps://qsofa.org/what.php, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5418298/), a flag (0 or 1) for determining if a patient has Sepsis based on qSOFA Score, and a flag for if the patient has Sepsis based on SOFA score.
### Histograms for Initial Data.ipynb
This file provides a brief analysis of the avaiable data, as well as a huge set of histograms to demonstrate feasible range
### Q-Learning Script.ipynb
This is the bread and butter of the project, the script that actually builds the MDPs as models and conducts Q-Learning. The output is a Sepsis determining model

