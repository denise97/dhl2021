# Master Project: Predictive Maintenance for Patients 

This project was conducted in the summer term 2021 at the "Digital Health - Connected Healthcare" chair of the Hasso Plattner Institute for Digital Engineering, University of Potsdam, Germany. For more information, see [project description](https://hpi.de/arnrich/teaching/master-projects/forecasting-alarms-in-intensive-care-units.html) and links in Wiki entry [Materials and Links](https://github.com/denise97/dhl2021/wiki/Materials-and-Links). 

## Folder Structure in Git Repository

In general, this repository contains only scripts (usually Jupyter notebooks or Python scripts), not datasets. Datasets are located on the Delab server in the `MPSS2021BA1` folder (usually CSV, Pickle or Parquet files) and are mostly documented in the Wiki entry [Excerpt of Produced Datasets](https://github.com/denise97/dhl2021/wiki/Excerpt-of-Produced-Datasets).

**A tabular overview of all individual scripts is provided in the [Script Register](./Script_Register.csv)**, including the script name and location as well as the respective input and output file(s).

The folder structure is as follows:

- [**`archive`**](./archive/)
  - Obsolete scripts that are currently no longer used in this form
- [**`mimic_alarm_data_generation`** ](./mimic_alarm_data_generation)
  - Scripts for generating alarm data based on cleaned chartevents data 
  - Necessary because alarm data are not part of the MIMIC-III data sets; later we decided that forecasting scripts perform alarm data generation by their own 
  - Note that alarm data generation takes place after data pre-processing (e.g. threshold cleaning impacts where alarms are identified)
- [**`mimic_alarm_violations_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts used to analyze the `alarm_violations.csv` data set by Jonas Chromik
  - Created to gain initial data insights and before starting work on data preprocessing
- [**`mimic_chartevents_analysis`**](./mimic_alarm_violations_analysis)
  - Scripts for analyzing MIMIC-III CHARTEVENTS table in all variants, e.g. cleaned, chunked or resampled
- [**`mimic_medication_analysis`**](./mimic_medication_analysis)
  - Scripts for analyzing MIMIC-III medication datasets
- [**`mimic_data_preprocessing`**](./mimic_data_preprocessing)
  - Scripts for preparing MIMIC-III data for analysis and prediction, i.e. cleaning, chunking and resampling CHARTEVENTS data
- [**`mimic_prediction_arima`**](./mimic_prediction_arima)
  - Scripts for time series forecasting with *Autoregressive Integrated Moving Average* (ARIMA) models in all variants, i.e. also ARIMAX
- [**`mimic_prediction_rnn`**](./mimic_prediction_rnn)
  - Scripts for time series forecasting with [*RNNModel* by Darts](https://unit8co.github.io/darts/generated_api/darts.models.rnn_model.html), i.e. with Vanilla RNNs, LSTMs and GRUs
  - Models are tested with non-scaled, min-max-scaled and standard score-scaled time series
- [**`mimic_prediction_tcn`**](./mimic_prediction_tcn)
  - Scripts for time series forecasting with [*TCNModel* by Darts](https://unit8co.github.io/darts/generated_api/darts.models.tcn_model.html)
- [**`paper_visualizations`**](./paper_visualizations)
  - Scripts for the creation of various plots for paper
- [**`templates_for_plots`**](./templates_for_plots)
  - General templates for creating charts such as box plots, histograms and time series plots

## Glossary of Terms

- **Vital Parameter Value Series** (short *value series*) refers to a series of measurement points for a specific vital parameter, e.g. the heart rate in beats per minute over a period of time.
- **Alarm Threshold Value** (short *threshold*) is a limit value the crossing of which triggers an alarm. There can be different **Threshold Types**, namely *high* and *low*. For example, if the set threshold value of type high is exceeded by the heart rate measured in the patient, an alarm is triggered indicating that the heart rate is too high. Similar to the vital parameters, the threshold values are available as series of data points, e.g. an **Alarm Threshold Value Series** (short *threshold series*) containing the alarm threshold value for a too high heart rate over a period of time.
- **ICU Stay** is the stay of a patient in an intensive care unit. Note that a patient may stay in the ICU more than once over time, i.e. there are more ICU stays than patients.
- **Chunk** refers to a section of an ICU stay measurement series. **Chunking** of an ICU stay measurement series into several shorter measurement series (chunks) aims at removing longer periods without measurement points. In the case of **Parameter-Specific Chunking**, this splitting of the measurement series is performed separately for each ICU stay/ parameter combination, whereas in the case of **Cross-Parameter Chunking**, the splitting is performed for all parameters of an ICU stay at the same points. In any case, there are at least as many chunks as ICU stays, usually more chunks than ICU stays. Cross-Parameter Chunking usually leads to more chunks than Parameter-Specific Chunking.
- **Revert local threshold swap** (cleaning step, see [create_clean_chartevents.ipynb](./mimic_data_preprocessing/create_clean_chartevents.ipynb)): In some cases, the thresholds appear to be exactly swapped, i.e. the threshold of type *high* appears to be actually the threshold of type *low* and vice versa. Such an apparent swap is usually not present over the entire period of the alarm threshold value series, but only at certain sections, i.e. locations. Accordingly, we call it a *local threshold swap*. Local threshold swaps are corrected by reversing the swap, i.e. by swapping the alarm threshold value of type *high* with that of type *low* at the affected locations.
- **Remove threshold values due to local overlap** (cleaning step, see [create_clean_chartevents.ipynb](./mimic_data_preprocessing/create_clean_chartevents.ipynb)): There are also cases where the alarm threshold value series of type *high* and *low* overlap, but are not exactly swapped (e.g. the alarm threshold value of the type *high* is set unreasonably low and falls below the alarm threshold value of type *low*, while the latter seems to have a correct value). Such an overlap is usually not present over the entire period of the alarm threshold value series, but only at certain sections, i.e. locations. Accordingly, we call it a *local overlap*. These cases cannot be corrected by swapping. Also, it is not possible (without a lot of effort) to determine whether only one of the two thresholds is unreasonably high/low or both. Therefore, the threshold values of both types (high and low) are removed at those locations where they overlap. Hence, we call this step *Remove threshold values due to local overlap*.
- **Triggered Alarms** (short *alarms*, see [generate_alarm_data.py](./mimic_alarm_data_generation/generate_alarm_data.py)) are generated if there is a value in a vital parameter value series of a specific ICU stay that is higher than the threshold of type high or lower than the threshold of type low which was set last.
- **Medication** refers to medical treatments recorded in `PRESCRIPTIONS.csv` and `INPUTEVENTS_MV.csv`. After the investigation of these tables, we added a flag to each alarm indicating whether a medication was given within a certain period of time around the alarm (1h before to 1h after the alarm,  see [03_integrate_med_flag.ipynb](./mimic_medication_analysis/03_integrate_med_flag.ipynb)). This flag is solely based on `inputevents_based_medications.parquet`.
- **Sampling Rate** describes the average of vital parameter measurements obtained in one hour for a specific vital parameter of an ICU Stay or a Chunk.
- **Timedelta to Previous Measurement**  forms the basis for deriving possible chunking rules. The timedelta always refers to the timestamp of the same ICU stay and the same vital parameter. The elapsed time between one measurement and the previous measurement is given in minutes.

