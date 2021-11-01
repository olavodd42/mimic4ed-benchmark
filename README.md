MIMIC-IV-ED Benchmark
=========================

Python workflow for generating benchmark datasets and machine learning models from the MIMIC-IV-ED database.

## Table of contents
* [General info](#general-info)
* [Structure](#structure)
* [Requirements and Setup](#requirements-and-setup)
* [Workflow](#workflow)
    1. [Benchmark Data Generation](#1-benchmark-data-generation)
    2. [Cohort Filtering](#2-cohort-filtering)
    3. [Outcome and Model Selection](#3-outcome-and-model-selection)
    4. [Model Evaluation](#4-model-evaluation)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

## General info

Clinical decisions in the emergency department play an important role in optimizing urgent patient care and scarce resources. And unsurprisingly, machine learning based clinical prediction models have been widely adopted in the field of emergency medicine.

In parellel to the rise of clinical prediction models, there has also been a rapid increase in adoption of Electronic Health records (EHR) for patient data. The Medical Information Mart for Intensive Care ([MIMIC)-IV]((https://physionet.org/content/mimiciv/1.0/)) and [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/) are examples of EHR databases that contain a vast amount of patient information.

There is therefore a need for publicly available benchmark datasets and models that allow researchers to produce comparable and reproducible results. 

For the previous iteration of the MIMIC database (MIMIC-III), several benchmark pipelines have published in [2019](https://github.com/YerevaNN/mimic3-benchmarks) and [2020](https://github.com/MLforHealth/MIMIC_Extract).

Here, we present a workflow that generates a benchmark dataset from the MIMIC-IV-ED database and constructs benchmark models for three ED-based prediction tasks.


## Structure

The structure of this repository is detailed as follows:

- `Benchmark_scripts/...` contains the scripts for benchmark dataset generation (master_data.csv).
- `Benchmark_scripts/...` contains the scripts for building the various task-specific benchmark models.
-  

## Requirements and Setup
MIMIC-IV-ED and MIMIC-IV databases are not provided with this repository and are **required** for this workflow. MIMIC-IV-ED can be downloaded from [https://physionet.org/content/mimic-iv-ed/1.0/](https://physionet.org/content/mimic-iv-ed/1.0/) and MIMIC-IV can be downloaded from [https://physionet.org/content/mimiciv/1.0/](https://physionet.org/content/mimiciv/1.0/).

***NOTE** It should be noted that upon downloading and extracting the MIMIC databases from their compressed files, the directory `/mimic-iv-ed-1.0/ed` should be moved/copied to the directory containing MIMIC-IV data `/mimic-iv-1.0`.

## Workflow

The following sub-sections describe the sequential modules within the MIMIC-IV-ED workflow and how the should ideally be run.

Prior to these steps, this repository, MIMIC-IV-ED and MIMIC-IV should be downloaded and set up locally. 

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py --mimic4_path {mimic4_path} --output_path {output_path} --icu_transfer_timerange {icu_transfer_timerange} --next_ed_visit_timerange {next_ed_visit_timerange}
~~~

**Arguements**:

- `mimic4_path` : Path to directory containing MIMIC-IV data. Refer to [Requirements and Setup](#requirements-and-setup) for details.
- `output_path ` : Path to output directory.
- `icu_transfer_timerange` : Timerange in hours for ICU transfer outcome. Default set to 12. 
- `next_ed_visit_timerange` : Timerange in days days for next ED visit outcome. Default set to 3.

**Output**:

`master_dataset.csv` output to `output_path`

**Details**:

The input `edstays.csv` from the MIMIC-IV-ED database is taken to be the root table, with `subject_id` as the unique identifier for each patient and `stay_id` as the unique identifier for each ED visit. This root table is then merged with other tables from the main MIMIC-IV database to capture an informative array of clinical variables for each patient.

A total of **81** variables are included in `master_dataset.csv` (Refer to Table 3 for full variable list).


### 2. Cohort Filtering, Data Processing
~~~
python data_general_processing.py --master_dataset_path {master_dataset_path} --output_path {output_path}
~~~

**Arguements**:

- `master_dataset_path` : Path to directory containing "master_dataset.csv".
- `output_path ` : Path to output directory.

**Output**:

`train.csv` and `test.csv` output to `output_path`

**Details**:

`master_dataset.csv` is first filtered to remnove pediatric subjects (Age < 18).

Outlier values in vital sign and lab test variables are then detected using an identical method to [Wang et al.](https://github.com/MLforHealth/MIMIC_Extract), with outlier thresholds defined previously by [Harutyunyan et al.](https://github.com/YerevaNN/mimic3-benchmarks) Outliers are then imputed with the neareset valid value.

The data is then split into `train.csv` and `test.csv` and clinical scores for each patient are then added as additonal variables.


### 3. Outcome Selection and Model evaluation

## Acknowledgements

## Citation


