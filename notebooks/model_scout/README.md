# AI Job Post Classifier — Model Selection Benchmark

## Overview

This project evaluates pretrained zero-shot classification models for classifying English technical job posts into predefined career categories.

The goal is not to train a model from scratch. Instead, the project follows a pretrained-first workflow:

1. Define a real classification task.
2. Search for suitable pretrained models.
3. Read and compare model cards.
4. Run the same evaluation set through multiple models.
5. Compare accuracy, latency, model size, and deployment complexity.
6. Select the best model based on practical engineering trade-offs.
7. Build and publish an interactive Gradio demo.

## Problem Statement

The system receives an English technical job description and predicts its most relevant career domain.

Supported categories:

- Data Engineering and Data Pipelines
- Artificial Intelligence and Machine Learning
- DevOps and Deployment Automation
- Cloud Infrastructure and Cloud Engineering
- Cybersecurity and Information Security
- Software and Application Development

### Example Input

```text
We are looking for an engineer with experience in Apache Airflow,
dbt, Spark, PostgreSQL, SQL, and data warehouse design.
```

### Expected Output

```text
Data Engineering and Data Pipelines
```

## Candidate Labels

The project uses descriptive labels because zero-shot models rely on the semantic meaning of each label.

```python
candidate_labels = [
    "data engineering and data pipelines",
    "artificial intelligence and machine learning",
    "devops and deployment automation",
    "cloud infrastructure and cloud engineering",
    "cybersecurity and information security",
    "software and application development",
]
```

Using descriptive labels is usually more reliable than using short labels such as `AI`, `Cloud`, or `Software`.

## Candidate Models

The current shortlist includes three pretrained zero-shot classification models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
3. `knowledgator/gliclass-small-v1.0-lw`

Each candidate represents a different engineering trade-off.

| Model | Role in Evaluation | Expected Strength |
|---|---|---|
| BART Large MNLI | Baseline | Strong, widely used English zero-shot model |
| DeBERTa v3 Base | Accuracy candidate | Strong NLI performance and potentially better classification quality |
| GLiClass Small | Efficiency candidate | Smaller, faster, and easier to deploy |

The final model will be selected only after all candidates are evaluated on the same dataset.

## Evaluation Plan

The models will be tested on the same manually prepared dataset of English technical job descriptions.

The evaluation dataset will contain balanced examples across all six categories.

Planned dataset structure:

- Data Engineering: 5–10 examples
- Artificial Intelligence / Machine Learning: 5–10 examples
- DevOps: 5–10 examples
- Cloud Engineering: 5–10 examples
- Cybersecurity: 5–10 examples
- Software Development: 5–10 examples

The expected dataset size is approximately 30–60 job descriptions.

## Evaluation Metrics

The following metrics will be recorded for each model:

- Accuracy
- Average inference latency
- Total evaluation time
- Model size
- Memory requirements
- CPU performance
- Ease of local execution
- Ease of deployment
- Dependency complexity
- Output confidence
- Error cases and ambiguous predictions

## Current Progress

Completed:

- Defined the project idea.
- Selected the six job categories.
- Chose descriptive candidate labels.
- Shortlisted three pretrained models.
- Downloaded `facebook/bart-large-mnli`.
- Successfully loaded the first model locally.
- Tested the model on a Data Engineering job description.
- Displayed predictions and confidence scores using a Pandas DataFrame.

Current first-model result:

- Predicted category: Data Engineering and Data Pipelines
- Confidence: approximately 92%

This result is only an initial test and is not yet a final evaluation.

## Initial Inference Workflow

The first model is loaded using the Hugging Face Transformers pipeline:

```python
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

classifier = pipeline(
    task="zero-shot-classification",
    model=MODEL_NAME
)
```

A job description is classified using:

```python
result = classifier(
    job_description,
    candidate_labels=candidate_labels
)
```

The returned result contains:

- `sequence`: the original job description
- `labels`: categories ordered from highest to lowest score
- `scores`: confidence scores corresponding to each category

## Result Display

The prediction output is converted into a Pandas DataFrame for clearer visualization.

```python
import pandas as pd

results_df = pd.DataFrame({
    "Category": result["labels"],
    "Confidence (%)": [
        round(score * 100, 2)
        for score in result["scores"]
    ]
})

results_df
```

The first row represents the model's top prediction.

## Problems Encountered

### 1. Slow BART Model Download

The `facebook/bart-large-mnli` model requires downloading a large weight file.

The main `model.safetensors` file is approximately 1.63 GB.

The initial download appeared to remain frozen for long periods during file reconstruction.

### 2. Hugging Face Xet Reconstruction Issue

The Hugging Face CLI used the `hf-xet` download system.

The network download speed was acceptable, but the local reconstruction process was extremely slow on Windows.

Example output:

```text
Reconstructing: 7.01 MB / 1.63 GB at approximately 13.2 kB/s
```

This made the download appear stuck even though some data was still being processed.

### 3. Unauthenticated Hugging Face Requests

The Hugging Face Hub displayed this warning:

```text
You are sending unauthenticated requests to the HF Hub.
```

This warning did not stop the download, but unauthenticated users may have lower rate limits.

A future improvement is to authenticate using a Hugging Face access token.

### 4. Windows Symlink Warning

The Hugging Face cache system displayed a warning because Windows symlink support was not enabled.

The cache still worked, but in degraded mode, which may require more disk space.

Possible solutions:

- Enable Windows Developer Mode.
- Run the terminal as administrator.
- Continue without symlinks if sufficient disk space is available.

### 5. Downloading the Entire Repository

Running:

```powershell
hf download facebook/bart-large-mnli
```

attempted to download multiple repository files and framework versions, producing a reported total of approximately 6.93 GB.

Only the PyTorch-compatible model weights, tokenizer, and configuration files were required.

A selective download command was used instead:

```powershell
hf download facebook/bart-large-mnli `
  config.json `
  model.safetensors `
  tokenizer.json `
  tokenizer_config.json `
  vocab.json `
  merges.txt `
  special_tokens_map.json
```

This successfully downloaded the required files.

### 6. Partial Download Cleanup

Interrupted downloads created incomplete files in the Hugging Face cache.

The local cache path was:

```text
C:\Users\user\.cache\huggingface\hub\models--facebook--bart-large-mnli
```

Incomplete cache files needed to be removed before retrying the download.

### 7. Notebook File Organization Issue

The notebook files were moved into dedicated folders so that each notebook could have its own README.

The intended structure is:

```text
notebooks/
├── daily_quote_sentiment/
│   ├── daily_quote_sentiment.ipynb
│   └── README.md
└── model_scout/
    ├── model_scout.ipynb
    └── README.md
```

During the move, an unsaved notebook appeared empty because the saved disk version did not contain the latest cells.

This highlighted the importance of saving notebooks before moving or renaming them.

## Planned Project Structure

```text
ai-engineer-starter-kit/
├── notebooks/
│   ├── daily_quote_sentiment/
│   │   ├── daily_quote_sentiment.ipynb
│   │   └── README.md
│   └── model_scout/
│       ├── model_scout.ipynb
│       └── README.md
├── data/
│   └── job_posts_evaluation.csv
├── results/
│   └── model_comparison.csv
├── app.py
├── inference.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Next Steps

1. Restore or rebuild the `model_scout.ipynb` notebook.
2. Test BART on examples from all six categories.
3. Create a balanced evaluation dataset.
4. Measure BART accuracy and latency.
5. Download and evaluate DeBERTa.
6. Install and evaluate GLiClass.
7. Save all results in a comparison table.
8. Analyze model errors and ambiguous cases.
9. Select the best model using accuracy and latency trade-offs.
10. Build a reusable `inference.py` module.
11. Build a Gradio demo.
12. Publish the project on GitHub.
13. Deploy the final demo on Hugging Face Spaces.

## Final Expected Deliverables

- Public GitHub repository
- Clean project README
- Model comparison report
- Evaluation dataset
- Accuracy and latency comparison table
- Selected model with clear justification
- Reusable inference module
- Gradio web demo
- Public Hugging Face Space link

## Key Learning Objective

The main learning objective of this project is not simply to run a pretrained model.

The project demonstrates the complete applied AI workflow:

```text
Define Task
→ Search Models
→ Read Model Cards
→ Run Inference
→ Build Evaluation Dataset
→ Measure Accuracy and Latency
→ Compare Trade-offs
→ Select the Best Model
→ Build and Deploy a Demo
```