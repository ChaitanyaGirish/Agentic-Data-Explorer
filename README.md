# Agentic-Data-Explorer
An automated, agent-driven system built using LangChain, OpenAI, and Python to explore any tabular dataset.
The tool validates the dataset, prevents information leakage, rejects unsupported data types, performs automated EDA, and recommends machine-learning models — all in a human-friendly conversational format.

## Rejects datasets that contain:
Time series columns
Images/videos
Date columns
Text-heavy columns (based on threshold)

Accepts only clean numeric and categorical data.

## Components
The proposed system contains three agents. 

### 1.Main Agent (main.py):
Handles user instructions
Loads the dataset path
Delegates tasks to sub-agents
Produces human-friendly responses

### 2.Data Verification Agent (subagent_1.py):
Loads CSV
Validates schema
Detects unsupported columns
Returns cleaned data and a structured summary

### 3.Model Selector Agent (subagent_2.py):
Generates statistical summaries
Detects missing values
Reports distributions
Flags potential data leakage
Suggests ML models

## Workflow
1. User provides the dataset path(filepath)
   Main agent reads the file, checks structure, and delegates tasks.

2. Data verification Agent
   Loads CSV, Validates each column and Rejects dataset if it contains unsupported formats 

3. Model Selector Agent
  Performs data pre-processing steps and Suggests ML models based on data types & target

Main Agent returns a final, human-like narrative report.
