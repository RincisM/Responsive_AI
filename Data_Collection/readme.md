<!DOCTYPE html>
<html>
<head>
<style>
  body {
    font-family: monospace;
    color: #333;
    line-height: 1.6;
    margin: 20px;
  }
  h1, h2, h3 {
    color: #007BFF;
  }
  code {
    font-family: 'Courier New', monospace;
    background-color: #f8f9fa;
    padding: 2px 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  pre {
    background-color: #f8f9fa;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 10px;
    overflow: auto;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 20px;
  }
  th, td {
    border: 1px solid #ddd
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
</style>
</head>
<body>

# Project: Mitigating Bias in Data Collection

## Project Website
[Check the Project Website Here](https://datacollectionbiasmitigation.netlify.app/)

**Done By:**
| Name            | Roll Number      | Contribution         |
|-----------------|------------------|-----------------------|
| sriramselva     | 2022179009       | selection_bias     |
| sharmila        | 2022179011       | measurement_bias   |
| keerthivasan    | 2022179038       | website_creation, c4_model |

## Table of Contents

- [Abstract](#abstract)
- [Method](#method)
  - [Problem](#problem)
  - [Proposal](#proposal)
    - [Dataset Used](#dataset-used)
    - [Data Preprocessing](#data-preprocessing)
  - [Bias Findings](#bias-findings)
  - [Bias Mitigation](#bias-mitigation)
  - [How to Run the Project](#how-to-run-the-project)
- [Result](#result)

# Abstract

This study addresses challenges in data collection, emphasizing the detection and mitigation of selection and measurement bias. Advanced visualization is used to uncover bias patterns, followed by the application of techniques like reweighing and resampling. These methods enhance reliability and fairness in large datasets, providing a practical framework for bias mitigation.

# Method

## Problem

Despite technological advancements, challenges like selection and measurement bias persist in data collection, compromising analytical integrity. This study explores these biases and their impact on predictive models, emphasizing the risks of overfitting and underfitting.

## Proposal

### Dataset Used

Utilizing data from the Adult Income and Employee datasets, the project explores income patterns from diverse demographic and employment perspectives. Thorough data preprocessing ensures data quality, including handling missing values, standardizing formats, and addressing outliers.

### Data Preprocessing

The dataset undergoes preprocessing to handle missing values, standardize formats, and address outliers, enhancing overall cleanliness for accurate analysis and bias detection.

## Bias Findings

The Bias Findings section employs advanced visual analysis techniques to investigate selection and measurement bias. Statistical measures identify distortions within the data, while a column-wise analysis provides a detailed examination of potential biases. The findings offer insights into dataset integrity.


## Bias Mitigation

The Bias Mitigation section proactively addresses the identified biases. Techniques, including resampling and adjustment of class weights, are applied to mitigate selection and measurement bias.

## How to Run the Project

To run the project, follow these steps:

1. **Clone the Repository:**

   - `git clone https://github.com/RincisM/Responsive_AI.git`
   - `cd Data_Collection`

2. **Run Jupyter Notebooks:**
   - Run `selection_bias.ipynb`
   - Run `measurement_bias.ipynb`

# Result

Following bias mitigation, the dataset is transformed to ensure fair treatment of all demographic groups. The improved dataset integrity enhances the reliability and fairness of subsequent analyses, contributing to the ethical use of data in exploring income-related patterns and disparities.

</body>
</html>
