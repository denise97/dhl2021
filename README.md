# Masterproject Predictive Maintenance for Patients

## Alarm Management in Intensive Care Units

### Materials

* [MIMIC-III](https://mimic.physionet.org/) has irregular vital parameters and alarm thresholds (but no alarm events)
  * **[MIMIC-III Clinical Database 1.4](https://physionet.org/content/mimiciii/1.4/)** (Denise & Marius have access)
* [HiRID](https://hirid.intensivecare.ai/) vital parameters with a high(er) time-resolution but no alarm data
  * **[HiRID, a high time-resolution ICU dataset 1.1.1](https://physionet.org/content/hirid/1.1.1/)** (Denise & Marius **do not** have access)
* [eICU CRD](https://eicu-crd.mit.edu/) periodic vital parameters (every 5 minutes) but also no alarm data
  * **[eICU Collaborative Research Database 2.0](https://physionet.org/content/eicu-crd/2.0/)** (Denise & Marius have access)
  * [eICU git repo](https://github.com/mit-lcp/eicu-code)
* GitLab Repos Jonas:
  * https://gitlab.hpi.de/jonas.chromik/mimic-alarms
  * https://gitlab.hpi.de/jonas.chromik/eicu-crd-forecasting
* Methods:
  * [TensorFlow tutorial on time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series), but
you might want to start simpler with...
  * ...(S)ARIMA(X) models, check out https://www.statsmodels.org/stable/index.html and http://alkaline-ml.com/pmdarima/
 
 ## Federated Learning for Predicting Complications After Surgery
 
* Methods:
  * TensorFlow Federated (https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification) or similar framework.
  * Simple Logistic Regression model vs. Neural Network
