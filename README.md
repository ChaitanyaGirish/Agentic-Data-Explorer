# Agentic-Data-Explorer
An automated, agent-driven system built using LangChain, OpenAI, and Python to explore any tabular dataset.
The tool validates the dataset, prevents information leakage, rejects unsupported data types, performs automated EDA, and recommends machine-learning models — all in a human-friendly conversational format.

## Rejects datasets that contain:
(a) Time series columns 
(b) Images/videos 
(c) Date columns 
(d) Text-heavy columns (based on threshold)  

Accepts only clean numeric and categorical data.

## Components
The proposed system contains three agents. 

### 1.Main Agent (main.py):
(a) Handles user instructions
(b) Loads the dataset path
(c) Delegates tasks to sub-agents
(d) Produces human-friendly responses

### 2.Data Verification Agent (subagent_1.py):
(a) Loads CSV
(b) Validates schema
(c) Detects unsupported columns
(d) Returns cleaned data and a structured summary

### 3.Model Selector Agent (subagent_2.py):
(a) Generates statistical summaries
(b) Detects missing values
(c) Reports distributions
(d) Flags potential data leakage
(e) Suggests ML models

## Workflow
1. User provides the dataset path(filepath)
   Main agent reads the file, checks structure, and delegates tasks.

2. Data verification Agent
   Loads CSV, Validates each column and Rejects dataset if it contains unsupported formats 

3. Model Selector Agent
  Performs data pre-processing steps and Suggests ML models based on data types & target

Main Agent returns a final, human-like narrative report.
