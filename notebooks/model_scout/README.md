# AI Job Post Classifier — Model Selection Benchmark

## Overview

This project evaluates pretrained zero-shot classification models for classifying English technical job posts into predefined career categories.

The goal is not to train a model from scratch. Instead, the project follows a pretrained-first workflow:

```text
Define a real classification task
        ↓
Search for suitable pretrained models
        ↓
Read and compare model cards
        ↓
Run the same evaluation dataset through multiple models
        ↓
Compare accuracy, latency, model size, and deployment complexity
        ↓
Select the best model based on engineering trade-offs
        ↓
Build and publish an interactive Gradio demo
```

---

## Problem Statement

The system receives an English technical job description and predicts its most relevant career domain.

### Supported Categories

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

---

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

---

## Candidate Models

The shortlist contains three pretrained zero-shot classification models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
3. `knowledgator/gliclass-small-v1.0-lw`

| Model | Role in Evaluation | Expected Strength |
|---|---|---|
| BART Large MNLI | Baseline | Strong and widely used English zero-shot classifier |
| DeBERTa v3 Base | Accuracy candidate | Strong NLI performance and potentially better classification quality |
| GLiClass Small | Efficiency candidate | Smaller, faster, and easier to deploy |

The final model will be selected only after all candidates are evaluated on the same dataset.

---

## Evaluation Dataset

The models are evaluated using the same manually prepared dataset of English technical job descriptions.

### Current Dataset Size

```text
12 job descriptions
2 examples per category
```

This dataset is suitable for an initial benchmark, but it will be expanded before the final evaluation.

### Planned Final Dataset

- Data Engineering: 5–10 examples
- Artificial Intelligence and Machine Learning: 5–10 examples
- DevOps: 5–10 examples
- Cloud Engineering: 5–10 examples
- Cybersecurity: 5–10 examples
- Software Development: 5–10 examples

Expected final size:

```text
30–60 job descriptions
```

### Dataset Location

```text
notebooks/model_scout/data/job_posts_evaluation.csv
```

---

## Evaluation Metrics

The following metrics are recorded for each model:

- Accuracy
- Average inference latency
- Total evaluation time
- Average confidence
- Model size
- Memory requirements
- CPU performance
- Ease of local execution
- Ease of deployment
- Dependency complexity
- Error cases and ambiguous predictions
- Local or hosted inference cost

---

## Current Progress

### Completed

- Defined the project idea.
- Selected six job categories.
- Created descriptive candidate labels.
- Shortlisted three pretrained models.
- Downloaded `facebook/bart-large-mnli`.
- Loaded BART locally using Hugging Face Transformers.
- Created an evaluation CSV file.
- Evaluated BART on 12 job descriptions.
- Measured prediction confidence and inference latency.
- Added predicted labels and correctness flags to a Pandas DataFrame.
- Created a separate notebook for DeBERTa evaluation.
- Prepared the second candidate model for testing.
- Documented model-download, cache, and notebook-organization issues.

### In Progress

- Evaluating DeBERTa on the same dataset.
- Recording DeBERTa accuracy, confidence, and latency.
- Preparing the third model evaluation.
- Expanding the evaluation dataset.
- Building the final comparison table.

---

## BART Baseline Result

The first completed benchmark used:

```text
facebook/bart-large-mnli
```

### Preliminary Result

```text
Correct predictions: 11
Wrong predictions: 1
Total examples: 12
Accuracy: 91.67%
```

This is a preliminary result because the evaluation dataset is still small.

A high score on 12 examples is encouraging, but it is not sufficient for a final model-selection decision. The dataset will be expanded to provide a more reliable comparison.

---

## Initial Inference Workflow

The BART model is loaded using the Hugging Face Transformers pipeline:

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
    candidate_labels=candidate_labels,
    multi_label=False
)
```

The returned result contains:

- `sequence`: the original job description
- `labels`: candidate categories ordered from highest to lowest score
- `scores`: confidence scores corresponding to each category

---

## Batch Evaluation Workflow

Each job description is passed through the model while measuring inference latency.

```python
import time

predictions = []
confidences = []
latencies = []

for job_description in evaluation_df["job_description"]:
    start_time = time.perf_counter()

    result = classifier(
        job_description,
        candidate_labels=candidate_labels,
        multi_label=False
    )

    latency = time.perf_counter() - start_time

    predictions.append(result["labels"][0])
    confidences.append(result["scores"][0] * 100)
    latencies.append(latency)
```

The results are then added to the evaluation DataFrame:

```python
evaluation_df["predicted_category"] = predictions
evaluation_df["confidence_percent"] = confidences
evaluation_df["latency_seconds"] = latencies

evaluation_df["is_correct"] = (
    evaluation_df["expected_category"]
    == evaluation_df["predicted_category"]
)
```

---

## Result Display

The benchmark output includes:

```text
job_description
expected_category
predicted_category
confidence_percent
latency_seconds
is_correct
```

Example:

```python
evaluation_df[
    [
        "job_description",
        "expected_category",
        "predicted_category",
        "confidence_percent",
        "latency_seconds",
        "is_correct",
    ]
]
```

Incorrect predictions can be inspected separately:

```python
evaluation_df[
    evaluation_df["is_correct"] == False
]
```

---

## Planned Comparison Table

| Model | Accuracy | Average Latency | Average Confidence | Model Size | Cost | Deployment Complexity |
|---|---:|---:|---:|---:|---|---|
| BART Large MNLI | 91.67% preliminary | To be recorded | To be recorded | Large | Local compute only | Medium |
| DeBERTa v3 Base | In progress | In progress | In progress | Medium | Local compute only | Medium |
| GLiClass Small | Not tested yet | Not tested yet | Not tested yet | Small | Local compute only | To be evaluated |

The final winner will not necessarily be the model with the highest accuracy.

The decision will also consider:

- Latency
- Model size
- Memory usage
- CPU performance
- Ease of installation
- Ease of deployment
- Dependency complexity
- Accuracy on ambiguous job descriptions

---

## Problems Encountered

### 1. Slow BART Model Download

The `facebook/bart-large-mnli` model requires downloading a large weight file.

The main `model.safetensors` file is approximately 1.63 GB. The initial download appeared to remain frozen for long periods during file reconstruction.

### 2. Hugging Face Xet Reconstruction Issue

The Hugging Face CLI used the `hf-xet` download system.

Example output:

```text
Reconstructing: 7.01 MB / 1.63 GB at approximately 13.2 kB/s
```

The network download was acceptable, but local reconstruction was slow on Windows.

### 3. Unauthenticated Hugging Face Requests

The Hub displayed:

```text
You are sending unauthenticated requests to the HF Hub.
```

The warning did not stop the download, but authentication may improve rate limits.

### 4. Windows Symlink Warning

The Hugging Face cache displayed a warning because Windows symlink support was not enabled.

Possible solutions:

- Enable Windows Developer Mode.
- Run the terminal as administrator.
- Continue without symlinks if sufficient disk space is available.

### 5. Downloading the Entire Repository

Running:

```powershell
hf download facebook/bart-large-mnli
```

attempted to download unnecessary framework files.

A selective download was used instead:

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

### 6. Partial Download Cleanup

Interrupted downloads created incomplete cache files at:

```text
C:\Users\user\.cache\huggingface\hub\models--facebook--bart-large-mnli
```

Incomplete files needed to be removed before retrying.

### 7. Notebook File Organization Issue

The notebooks were moved into dedicated folders.

```text
notebooks/
├── daily_quote_sentiment/
│   ├── daily_quote_sentiment.ipynb
│   └── README.md
└── model_scout/
    ├── model_scout.ipynb
    ├── DeBERTa.ipynb
    ├── data/
    │   └── job_posts_evaluation.csv
    └── README.md
```

An unsaved notebook appeared empty after moving because the latest cells had not been saved to disk.

### 8. Independent Notebook State

The DeBERTa evaluation uses a separate notebook.

Each notebook has its own kernel state, so variables such as:

```text
candidate_labels
evaluation_df
deberta_classifier
```

must be defined again.

### 9. Repeated Evaluation Lists

If an evaluation loop is executed more than once without resetting the result lists, predictions may be duplicated.

```python
deberta_predictions = []
deberta_confidences = []
deberta_latencies = []
```

These lists should be reset before each new run.

---

## Current Project Structure

```text
ai-engineer-starter-kit/
├── notebooks/
│   ├── daily_quote_sentiment/
│   │   ├── daily_quote_sentiment.ipynb
│   │   └── README.md
│   └── model_scout/
│       ├── model_scout.ipynb
│       ├── DeBERTa.ipynb
│       ├── data/
│       │   └── job_posts_evaluation.csv
│       └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

### Planned Additions

```text
results/
└── model_comparison.csv

inference.py
app.py
evaluate.py
```

---

## Next Steps

1. Complete the DeBERTa evaluation.
2. Record DeBERTa accuracy, latency, and average confidence.
3. Inspect DeBERTa error cases.
4. Evaluate `knowledgator/gliclass-small-v1.0-lw`.
5. Expand the dataset from 12 to at least 30 examples.
6. Re-run all three models on the expanded dataset.
7. Save the benchmark results in a comparison table.
8. Analyze ambiguous and incorrect predictions.
9. Select the final model using practical trade-offs.
10. Create a reusable `inference.py` module.
11. Build a Gradio demo.
12. Deploy the demo on Hugging Face Spaces.
13. Add the public Space link to this README.
14. Update the root repository README.

---

## Final Expected Deliverables

- Public GitHub repository
- Clean project README
- Model-card comparison
- Evaluation dataset
- Three evaluated pretrained models
- Accuracy comparison
- Latency comparison
- Cost and deployment comparison
- Error analysis
- Selected model with clear justification
- Reusable inference module
- Gradio web demo
- Public Hugging Face Space link

---

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
→ Compare Engineering Trade-offs
→ Select the Best Model
→ Build and Deploy a Demo
```

The most important skill demonstrated by this project is engineering judgment: selecting the model that provides the best practical balance between quality, speed, cost, and deployability.