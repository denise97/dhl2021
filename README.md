# Master Project: Predictive Maintenance for Patients 

For more information, see [project description](https://hpi.de/arnrich/projects/preventive-maintenance-for-patients.html) and links in Wiki entry [Materials](https://github.com/denise97/dhl2021/wiki/Material). 

## Folder Structure of Git Repository

In general, this Git repository contains only scripts (usually Jupyter notebooks or Python scripts), not data sets. Data sets are located on the Delab Summon server in the `MPSS2021BA1` folder (usually CSV or Parquet files).

**A tabular overview of all individual scripts is provided in the [Script Register](./Script_Register.csv)**, including the script name and location as well as the respective input and output file(s).

The folder structure is as follows:

- [**`archive`**](./archive/)
  - Obsolete scripts that are currently no longer used in this form.
- [**`mimic_alarm_violations_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts used to analyze the `alarm_violations.csv` data set by Jonas.
  - Created to gain initial data insights in Week02 of the project, see presentation slides `Weekly_02.pptx` from 2021-04-20 on [shared OneDrive cloud folder](https://onedrive.live.com/?id=EA16765E72B7F0C0%21441510&cid=EA16765E72B7F0C0)).
  - Created before starting work on data pre-processing.
- [**`mimic_chartevents_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts for analyzing MIMIC chartevent datasets in all variants, i.e. of cleaned chartevents, of chunked chartevents, of resampled chartevents, etc.
  - This folder may be further structured by subfolders. For example, a dedicated subfolder for sample rate analysis may be created if there are too many individual scripts for sampling rate analysis of ICU stays, of chunks, etc.
- [**`mimic_data_preprocessing`**](./mimic_data_preprocessing)
  - Scripts for preparing MIMIC data for analysis and prediction, e.g. cleaning, chunking and resampling CHARTEVENT data.
  - Once the order of pre-processing steps is established (e.g., Cleaning → Chunking → Resampling), we may add prefixes to the file names to reflect this sequences (e.g., `01_cleaning_...`, `02_chunking_...`, `03_resampling_...`). Currently only a proposal, not yet implemented.
- [**`mimic_prediction_arima`**](./mimic_prediction_arima)
  - Scripts for time series forecasting with *Autoregressive Integrated Moving Average models* (ARIMA) in all variants, i.e. also ARIMAX, for instance.
- [**`mimic_prediction_rnn`**](./mimic_prediction_rnn)
  - Scripts for time series prediction using *Recurrent Neural Networks* (RNNs).
- [**`templates_for_plots`**](./templates_for_plots)
  - General templates for creating charts such as box plots, histograms and time series plots.

##