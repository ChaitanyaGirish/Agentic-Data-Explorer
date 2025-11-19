# Agentic-Data-Explorer
An automated, agent-driven system built using LangChain, OpenAI, and Python to explore any tabular dataset.
The tool validates the dataset, prevents information leakage, rejects unsupported data types, performs automated EDA, and recommends machine-learning models — all in a human-friendly conversational format.

## Rejects datasets that contain:
(a) Time series columns <br>
(b) Images/videos <br>
(c) Date columns <br>
(d) Text-heavy columns (based on threshold)  <br>

Accepts only clean numeric and categorical data.

## Components
The proposed system contains three agents. 

### 1.Main Agent (main.py):
(a) Handles user instructions <br>
(b) Loads the dataset path <br>
(c) Delegates tasks to sub-agents <br>
(d) Produces human-friendly responses <br><br>

### 2.Data Verification Agent (subagent_1.py):
(a) Loads CSV <br>
(b) Validates schema <br>
(c) Detects unsupported columns <br>
(d) Returns cleaned data and a structured summary <br><br>

### 3.Model Selector Agent (subagent_2.py):
(a) Generates statistical summaries <br>
(b) Detects missing values <br>
(c) Reports distributions <br>
(d) Flags potential data leakage <br>
(e) Suggests ML models <br><br>

## Workflow
1. User provides the dataset path(filepath)<br>
   Main agent reads the file, checks structure, and delegates tasks.<br><br>

2. Data verification Agent<br>
   Loads CSV, Validates each column and Rejects dataset if it contains unsupported formats <br><br>

3. Model Selector Agent<br>
  Performs data pre-processing steps and Suggests ML models based on data types & target<br><br>

Main Agent returns a final, human-like narrative report.
