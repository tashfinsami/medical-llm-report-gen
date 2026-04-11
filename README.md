# Medical Imaging → Structured Report Generation (LLM Pipeline)

## Overview
This project implements an end-to-end pipeline that converts **mammogram images** into structured diagnostic reports using Large Language Models (LLMs). The system is designed to reflect real-world clinical workflows with an emphasis on modularity, interpretability, and evaluation.

The report generation component uses Qwen3 as the underlying language model.

---

## Pipeline Architecture

The pipeline consists of three main stages:

### 1. Feature Extraction
- Extracts clinically relevant features from **mammogram images**  
- Includes detection of:
  - Mass characteristics (e.g., irregular, spiculated)
  - Calcification patterns (e.g., pleomorphic, segmental)
  - Architectural distortion  
- Produces structured intermediate representations for downstream processing  

---

### 2. Report Generation
- Uses Qwen3 via OpenAI Chat Completions API to convert extracted features into structured diagnostic reports  
- Ensures consistent formatting and medically relevant terminology  
- Focuses on:
  - Reducing hallucinations  
  - Maintaining factual alignment with extracted features  

---

### 3. Evaluation
- Implements multiple evaluation strategies:
  - Ground truth comparison (e.g., malignancy labels)  
- Analyzes:
  - Consistency   
  - Hallucination behavior  

---

## Prompt Engineering
- Designed structured prompts to enforce consistent medical reporting format  
- Used explicit constraints to prevent unsupported findings  
- Incorporated feature-grounded inputs to guide generation  
- Reduced hallucination by limiting generation scope to extracted features  

---

## Function Calling / Structured Output
- Implemented function-calling style outputs for structured report generation  
- Enforced schema-based responses (e.g., Mass, Calcifications, Distortion)  
- Improved reliability and downstream evaluation by avoiding free-form text  
- Enabled easier parsing and validation of model outputs  

---

## Tech Stack
- Python  
- Qwen3 VL 235B A22B Thinking
- OpenAI Chat Completions API
- Image processing libraries  
- Jupyter Notebooks  

---

## Project Structure

```bash
medical-llm-report-gen/
│
├── notebooks/
├── src/
│   ├── feature_extraction.py
│   ├── report_generation.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── outputs/
├── README.md
└── requirements.txt
```

---

## Example Output
Input: Mammogram image  

Output:
- Mass: Present, irregular, spiculated  
- Calcifications: Pleomorphic, segmental  
- Architectural Distortion: Present  

---

## Key Contributions
- Designed a modular pipeline for **mammogram analysis** with feature extraction, generation, and evaluation  
- Applied structured prompt engineering for controlled medical text generation  
- Implemented function-calling based structured outputs  
- Developed evaluation strategies to analyze hallucination and output quality  

---

## Challenges
- Controlling hallucinations in medical text generation  
- Limited availability of labeled mammogram data  
- Ensuring clinically meaningful structured outputs  

---

## Future Work
- Integrate vision-language models (VLMs)  
- Improve evaluation using domain-specific clinical metrics
- Rule based validation for required findings  
- LLM-as-a-judge scoring 
- Deploy as an interactive diagnostic support tool  

---

## Note
This project uses non-sensitive / sample data for demonstration purposes only.

---

## Author
Sk. Md. Tashfin Sami
