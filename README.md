# Master Project: Predictive Maintenance for Patients 

For more information, see [project description](https://hpi.de/arnrich/projects/preventive-maintenance-for-patients.html) and links in Wiki entry [Materials](https://github.com/denise97/dhl2021/wiki/Material). 

## Folder Structure in Git Repository

In general, this Git repository contains only scripts (usually Jupyter notebooks or Python scripts), not data sets. Data sets are located on the Delab Summon server in the `MPSS2021BA1` folder (usually CSV or Parquet files).

**A tabular overview of all individual scripts is provided in the [Script Register](./Script_Register.csv)**, including the script name and location as well as the respective input and output file(s).

The folder structure is as follows:

- [**`archive`**](./archive/)
  - Obsolete scripts that are currently no longer used in this form.
- [**`mimic_alarm_data_generation`** ](./mimic_alarm_data_generation)
  - Scripts for generating alarm data based on chartevents data. Necessary because alarm data are not part of the MIMIC data sets. Note that alarm alarm data generation takes place after data pre-processing (e.g. threshold cleaning impacts where alarms are identified).
- [**`mimic_alarm_violations_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts used to analyze the `alarm_violations.csv` data set by Jonas.
  - Created to gain initial data insights in Week02 of the project, see presentation slides `Weekly_02.pptx` from 2021-04-20 on [shared OneDrive cloud folder](https://onedrive.live.com/?id=EA16765E72B7F0C0%21441510&cid=EA16765E72B7F0C0)).
  - Created before starting work on data pre-processing.
- [**`mimic_chartevents_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts for analyzing MIMIC chartevent datasets in all variants, i.e. of cleaned chartevents, of chunked chartevents, of resampled chartevents, etc.
  - This folder may be further structured by subfolders. For example, a dedicated subfolder for sample rate analysis may be created if there are too many individual scripts for sampling rate analysis of ICU stays, of chunks, etc.
- [**`mimic_medication_analysis`**](./mimic_medication_analysis)
  - Scripts for analyzing MIMIC medication datasets.
- [**`mimic_data_preprocessing`**](./mimic_data_preprocessing)
  - Scripts for preparing MIMIC data for analysis and prediction, e.g. cleaning, chunking and resampling CHARTEVENT data.
  - Once the order of pre-processing steps is established (e.g., Cleaning → Chunking → Resampling), we may add prefixes to the file names to reflect the order (e.g., `01_cleaning_...`, `02_chunking_...`, `03_resampling_...`). Currently only a proposal, not yet implemented.
- [**`mimic_prediction_arima`**](./mimic_prediction_arima)
  - Scripts for time series forecasting with *Autoregressive Integrated Moving Average models* (ARIMA) in all variants, i.e. also ARIMAX, for instance.
- [**`mimic_prediction_rnn`**](./mimic_prediction_rnn)
  - Scripts for time series prediction using *Recurrent Neural Networks* (RNNs).
  - Folder does not exist yet, because there are no corresponding scripts yet.
- [**`templates_for_plots`**](./templates_for_plots)
  - General templates for creating charts such as box plots, histograms and time series plots.

## Glossary of Terms

- **Vital Parameter Value Series** (short *value series*) refers to a series of measurement points for a specific vital parameter, e.g. the heart rate in beats per minute over a period of time.
- **Alarm Threshold Value** (short *threshold*) is a limit value the crossing of which triggers an alarm. There can be different **Threshold Types**, namely *high* and *low*. For example, if the set threshold value of type high is exceeded by the heart rate measured in the patient, an alarm is triggered indicating that the heart rate is too high. Similar to the vital parameters, the threshold values are available as series of data points, e.g. an **Alarm Threshold Value Series** (short *threshold series*) containing the alarm threshold value for a too high heart rate over a period of time.
- **ICU Stay** is the stay of a patient in an intensive care unit. Note that a patient may stay in the ICU more than once over time, i.e. there are more ICU stays than patients.
- **Chunk** refers to a section of an ICU stay measurement series. **Chunking** of an ICU stay measurement series into several shorter measurement series (chunks) aims at removing longer periods without measurement points. In the case of **Parameter-Specific Chunking**, this splitting of the measurement series is performed separately for each ICU stay/ parameter combination, whereas in the case of **Cross-Parameter Chunking**, the splitting is performed for all parameters of an ICU stay at the same points. In any case, there are at least as many chunks as ICU stays, usually more chunks than ICU stays. Cross-Parameter Chunking usually leads to more chunks than Parameter-Specific Chunking.
- **Local Threshold Removal** ... *description to be added* 
- **Local Threshold Swap** ... *description to be added* 
- **Triggered Alarm** ... *description to be added* 
- **Sampling Rate** describes the average of vital parameter measurements obtained in one hour for a specific vital parameter of an ICU Stay or a Chunk.
- **Timedelta to Previous Measurement**  forms the basis for deriving possible chunking rules. The timedelta always refers to the timestamp of the same ICU stay and the same vital parameter. The elapsed time between one measurement and the previous measurement is given in minutes.
-  ... *list to be extended*
