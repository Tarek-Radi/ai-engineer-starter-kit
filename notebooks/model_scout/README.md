# AI Job Post Classifier — Model Selection Benchmark

## Live Demo

Hugging Face Space:

https://huggingface.co/spaces/TarekRadii/model-scout-comparison

Direct application:

https://tarekradii-model-scout-comparison.hf.space

---

## Overview

This project evaluates three pretrained zero-shot classification models for classifying English technical job posts into predefined career categories.

The goal is not to train or fine-tune a model. Instead, the project follows a pretrained-first workflow:

```text
Define a real classification task
        ↓
Search for suitable pretrained models
        ↓
Read and compare model cards
        ↓
Evaluate every model on the same dataset
        ↓
Measure accuracy, confidence, latency, and cost
        ↓
Compare engineering trade-offs
        ↓
Select a practical winner
        ↓
Publish an interactive Gradio dashboard
```

The main skill demonstrated by this project is engineering judgment: selecting a model based on measurable quality, speed, cost, and deployability rather than choosing a model by popularity alone.

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

Zero-shot classification depends on the semantic meaning of the candidate labels, so the project uses descriptive labels instead of short names such as `AI`, `Cloud`, or `Software`.

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

---

## Evaluated Models

| Model | Hugging Face Model ID | Role |
|---|---|---|
| BART Large MNLI | `facebook/bart-large-mnli` | Strong and widely used zero-shot baseline |
| DeBERTa v3 Base | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Accuracy-focused candidate |
| ModernBERT Base NLI | `tasksource/ModernBERT-base-nli` | Speed and efficiency candidate |

All three models were tested using the same candidate labels and the same evaluation dataset.

---

## Evaluation Dataset

The benchmark uses a manually prepared CSV file containing English technical job descriptions and their expected categories.

### Dataset Size

```text
12 job descriptions
2 examples per category
6 categories
```

### Dataset Schema

```text
job_description, expected_category
```

### Dataset Location

```text
notebooks/model_scout/data/job_posts_evaluation.csv
```

Example:

```csv
job_description,expected_category
"We need an engineer to build ETL pipelines using Python, SQL, Airflow, dbt, and Spark.","data engineering and data pipelines"
```

---

## Evaluation Methodology

Each model receives:

- The same 12 job descriptions
- The same six candidate labels
- `multi_label=False`
- The same expected categories
- The same evaluation logic

For every prediction, the project records:

- Predicted category
- Confidence score
- Inference latency
- Whether the prediction is correct

Before accuracy is calculated, expected and predicted labels are normalized using lowercase text and whitespace removal. This prevents formatting differences such as the following from being counted as errors:

```text
Data Engineering and Data Pipelines
data engineering and data pipelines
```

### Accuracy

```text
Accuracy = Correct Predictions / Total Examples
```

### Average Latency

```text
Average Latency = Total Inference Time / Number of Examples
```

### Cost

All models were executed locally using open pretrained weights.

```text
API cost: $0.00
Cost description: Free — local inference
```

---

## Benchmark Results

| Model | Accuracy | Correct | Wrong | Average Latency | Average Confidence | Cost |
|---|---:|---:|---:|---:|---:|---:|
| BART Large MNLI | 91.67% | 11 | 1 | Stored in final comparison CSV | Stored in final comparison CSV | $0.00 |
| DeBERTa v3 Base | 91.67% | 11 | 1 | 0.2268 s | 78.77% | $0.00 |
| ModernBERT Base NLI | 83.33% | 10 | 2 | 0.0977 s | 41.82% | $0.00 |

The complete comparison is stored in:

```text
notebooks/model_scout/data/final_result_of_comparison.csv
```

The deployed Gradio dashboard reads this CSV directly and displays the latest recorded values.

---

## Selected Model

### Current Winner: BART Large MNLI

BART and DeBERTa achieved the same accuracy:

```text
91.67%
```

The dashboard ranks models using:

1. Higher accuracy
2. Lower latency as the tie-breaker

Under this ranking strategy, BART is displayed as the current winner.

### Engineering Trade-off

- **BART** provides the best current balance under the accuracy-first ranking.
- **DeBERTa** achieved the same accuracy and the highest recorded average confidence among the fully measured candidates.
- **ModernBERT** was the fastest model at approximately `0.0977` seconds per prediction, but its accuracy was lower.

This demonstrates why model selection should not rely on one metric alone.

---

## Error Analysis

ModernBERT correctly classified 10 of the 12 examples.

Its two incorrect predictions involved categories with overlapping technical language:

1. A RAG-related role was predicted as Software Development instead of Artificial Intelligence and Machine Learning.
2. A secure Azure environment role was predicted as Cybersecurity instead of Cloud Infrastructure and Cloud Engineering.

These errors show that categories such as:

- AI vs. Software Development
- Cloud vs. Cybersecurity
- Cloud vs. DevOps

can overlap semantically and require a larger, more diverse evaluation dataset.

---

## Inference Example

```python
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

classifier = pipeline(
    task="zero-shot-classification",
    model=MODEL_NAME
)

result = classifier(
    "Build ETL pipelines using Airflow, dbt, Spark, and PostgreSQL.",
    candidate_labels=candidate_labels,
    multi_label=False
)

print(result["labels"][0])
print(result["scores"][0])
```

---

## Batch Evaluation Example

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

The predictions are then compared with the expected categories:

```python
evaluation_df["predicted_category"] = predictions
evaluation_df["confidence_percent"] = confidences
evaluation_df["latency_seconds"] = latencies

evaluation_df["expected_normalized"] = (
    evaluation_df["expected_category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

evaluation_df["predicted_normalized"] = (
    evaluation_df["predicted_category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

evaluation_df["is_correct"] = (
    evaluation_df["expected_normalized"]
    == evaluation_df["predicted_normalized"]
)
```

---

## Gradio Dashboard

The project includes a Gradio application that reads the final comparison CSV and presents:

- Accuracy winner
- Fastest model
- Number of evaluated models
- Accuracy comparison
- Latency comparison
- Average confidence comparison
- Correct and incorrect prediction counts
- Cost information
- Recommended model
- Downloadable comparison table

The application is deployed publicly on Hugging Face Spaces.

---

## Project Structure

```text
ai-engineer-starter-kit/
├── notebooks/
│   ├── daily_quote_sentiment/
│   │   ├── daily_quote_sentiment.ipynb
│   │   └── README.md
│   │
│   └── model_scout/
│       ├── app.py
│       ├── BART Model.ipynb
│       ├── DeBERTa.ipynb
│       ├── ModernBERT.ipynb
│       ├── README.md
│       └── data/
│           ├── job_posts_evaluation.csv
│           └── final_result_of_comparison.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

The evaluation notebooks are kept separate because each notebook has its own kernel state and model-specific workflow.

---

## Local Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

For the deployed comparison dashboard, the minimum dependencies are:

```text
gradio
pandas
```

The model-evaluation notebooks also use:

```text
torch
transformers
huggingface_hub
```

---

## Run the Gradio Application Locally

From the repository root:

```powershell
python notebooks/model_scout/app.py
```

The application will be available locally at:

```text
http://127.0.0.1:7860
```

---

## Deployment

The dashboard is deployed on Hugging Face Spaces using:

```text
SDK: Gradio
Application file: app.py
Comparison data: data/final_result_of_comparison.csv
```

The deployment Space contains:

```text
app.py
requirements.txt
README.md
data/
└── final_result_of_comparison.csv
```

The deployed dashboard does not run the three large models. It displays the benchmark results already saved in the comparison CSV, which keeps deployment lightweight and avoids unnecessary runtime cost.

---

## Model Cards and Responsible Use

Before evaluation, each candidate model should be reviewed through its Hugging Face model card.

Important information includes:

- Intended use
- Base model
- Training datasets
- Supported languages
- License
- Limitations
- Biases
- Evaluation results
- Deployment requirements

This project is educational and intended for technical job-post categorization.

It should not be used as the sole basis for:

- Hiring decisions
- Candidate rejection
- Employee assessment
- Other high-impact employment decisions

Human review is required when classification results may affect real people.

---

## Limitations

- The current dataset contains only 12 examples.
- Each category has only two examples.
- Confidence scores from different models are not always directly calibrated.
- Latency depends on the hardware, installed libraries, cache state, and whether CPU or GPU is used.
- Technical job descriptions often contain skills from more than one category.
- The current benchmark evaluates single-label classification only.
- Results should be re-evaluated on a larger and more diverse dataset before production use.

---

## Future Improvements

- Expand the evaluation dataset to 30–60 job descriptions.
- Add more ambiguous and mixed-domain examples.
- Record model download size and memory usage.
- Add macro F1-score and a confusion matrix.
- Test CPU-only deployment performance.
- Add automated benchmark scripts.
- Add per-category accuracy.
- Add model-card comparison notes.
- Add an optional live classification interface.
- Re-run the benchmark after dataset expansion.

---

## Final Deliverables

- Three evaluated pretrained models
- Shared evaluation dataset
- Accuracy comparison
- Latency comparison
- Confidence comparison
- Cost comparison
- Error analysis
- Final comparison CSV
- Selected model with engineering rationale
- Interactive Gradio dashboard
- Public Hugging Face Space
- Documented project workflow

---

## Key Learning Outcome

This project demonstrates the complete applied AI model-selection workflow:

```text
Define Task
→ Search Models
→ Read Model Cards
→ Run Inference
→ Build Evaluation Dataset
→ Measure Accuracy and Latency
→ Compare Engineering Trade-offs
→ Select a Model
→ Build a Demo
→ Deploy Publicly
```

The most important result is not simply identifying the highest-scoring model.

The project demonstrates how to make and justify a practical engineering decision using measurable evidence.