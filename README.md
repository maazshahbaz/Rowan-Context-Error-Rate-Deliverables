# FAA Context Error Rate

This repository contains code for computing the **FAA Context Error Rate (CER)** — a domain-specific metric designed to assess the **operational impact** of transcription errors in Air Traffic Control (ATC) communication. In addition to our CER implementation, we include a prompt-tuned LLM evaluation pipeline that provides detailed scoring and categorization of transcription errors based on their operational significance.

For the CER, we are using a **Meta LLaMA-3 70B Instruct** model and few-shot prompting. The system compares **reference transcriptions** with **ASR-generated outputs** to evaluate if transcription differences affect critical ATC semantics such as altitudes, headings, or instructions.

---

## 📖 Background (For Non-Technical Users)

In Air Traffic Control (ATC), accurate communication is critical. Traditional metrics like **Word Error Rate (WER)** only count how many words are wrong, but they don't tell you **if those mistakes actually matter**.

For example:
- Mishearing "**five thousand**" as "**four thousand**" is dangerous.
- Mishearing "**the**" as "**a**" doesn't matter.

The **FAA Context Error Rate (CER)** focuses on identifying **operationally significant errors** — ensuring that ASR (Automatic Speech Recognition) systems are evaluated based on what truly impacts flight safety.

This tool uses AI (a large language model) to **analyze transcription errors in context**.

---

## 📑 Table of Contents

- [Overview](#overview)  
- [Prerequisites (Non-Technical)](#prerequisites-non-technical)  
- [Model Setup and Configuration](#model-setup-and-configuration)  
- [Evaluation Methodology](#evaluation-methodology)  
- [Files Overview](#files-overview)  
- [Instructions for Use](#instructions-for-use)  
- [Sample Output](#sample-output)  
- [Dependencies](#dependencies)  
- [Credits](#credits)  

---

## Overview

Traditional Word Error Rate (WER) metrics may penalize harmless errors or fail to capture critical miscommunications. The **FAA Context Error Rate** goes beyond WER by focusing on:

- **Contextual Equivalence**: Does the transcription preserve operational intent?
- **Operationally Significant Errors**: Are there any miscommunications in altitude, commands, callsigns, or navigation?
- **Explanation of Impact**: Why the difference matters — or why it doesn't.

The evaluation is powered by a **quantized LLM pipeline** optimized for large-scale analysis.

---



## Prerequisites (Non-Technical)

### What is a Virtual Environment?
A **virtual environment** is an isolated space on your computer where you can install Python packages without affecting the rest of your system. It keeps dependencies for different projects separate.

### What is Hugging Face?
[Hugging Face](https://huggingface.co/) is an online platform hosting machine learning models and datasets. We use it to download and run the **Meta LLaMA-3 70B Instruct** model.

### How to Obtain Your Hugging Face Access Token
1. Sign up or log in at https://huggingface.co/.  
2. Click your profile → **Settings** → **Access Tokens**.  
3. Create a **New Token** (give it a name like `faa-cer-token`).  
4. Copy the token string — you’ll paste it into your code in the next steps.

   ## do something similar: In the cer.py file, paste your access token

```python
access_token = "paste_your_token_here"
```

---


## Model Setup and Configuration

This system uses:

- **Meta LLaMA-3 70B Instruct** via Hugging Face  
- **8-bit quantization** (BitsAndBytes) for efficient memory usage  
- `transformers.pipeline` for text generation  
- LangChain's `LLMChain` for structured prompt execution  

The model automatically distributes across available GPUs via `device_map="auto"`.

---

## Evaluation Methodology

The prompt uses **few-shot learning** with ATC-specific examples to guide the model in determining:

1. **Contextual Status**  
    - `"Contextually Equivalent"`: Differences that do not affect operational meaning.  
    - `"Contextually Different"`: Changes that could impact flight safety or intent.

2. **Errors Detected**  
    - A concise list of critical deviations.

3. **Explanation**  
    - A short rationale for why these differences matter.

---

## Files Overview

| File/Directory                      | Description                                                          |
|-------------------------------------|----------------------------------------------------------------------|
| `README.md`                         | Main documentation for CER implementation and usage instructions      |
| `cer.py`                            | Main script to load input JSON, run evaluations, and output results  |
| `Scoring based LLM Evaluation/`     | Additional implementation for scoring-based ASR evaluation           |
| ├── `Evaluation_pipeline.py`        | Implementation using Qwen3-30B-A3B for detailed error categorization |
| ├── `README.md`                     | Documentation for the scoring-based evaluation approach              |
| └── `flowchart.png`                 | Visual representation of the evaluation pipeline                      |

---

## Instructions for Use

### 1. Install Dependencies

Use `pip` to install the required packages:

```bash
pip install torch transformers langchain accelerate bitsandbytes
```

Login to Hugging Face:

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

---

### 2. Input Format

Your input JSON file should follow this format:

```json
[
  {
    "reference": "american two zero three climb and maintain eight thousand",
    "transcription": "american two zero three climb and maintain eight thousand"
  }
]
```

---

### 3. Run the Script

To evaluate the contextual error rate, simply run:

```bash
python cer.py
```

The script will read your input JSON, evaluate each transcription pair using the LLM, and output a new file containing:

- `"Contextual Status"`  
- `"Errors Detected"`  
- `"Explanation"`

---

## Sample Output

```json
{
  "reference": "delta five five six descend to four thousand",
  "transcription": "delta five five six descend to five thousand",
  "Contextual Status": "Contextually Different",
  "Errors Detected": "Numeric value changed (from four thousand to five thousand)",
  "Explanation": "The altitude difference may affect the intended operational instructions."
}
```


## Credits

This FAA Context Error Rate tool was developed as part of the FAA’s research into reliable automatic speech recognition (ASR) systems for ATC environments.

Previous deliverable:  
➡️ Rowan Speech-to-Text Fine-Tuning Deliverable 5
