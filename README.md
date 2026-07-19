# AI Engineer Starter Kit

A growing collection of hands-on AI engineering projects focused on pretrained models, external APIs, model evaluation, reusable inference pipelines, and practical deployment.

This repository documents my progress through an industry-oriented AI engineering roadmap. Each project is organized in its own notebook folder with dedicated documentation, while this root README provides an overview of the complete repository.

---

## Repository Goals

The main goals of this repository are to:

- Learn AI engineering through practical projects.
- Use pretrained models instead of training everything from scratch.
- Integrate external APIs with AI workflows.
- Build clean and reproducible Jupyter Notebooks.
- Compare models using measurable engineering criteria.
- Organize inference code into reusable components.
- Document technical decisions, problems, and solutions.
- Build portfolio-ready AI projects.
- Progress toward interactive demos and public deployments.

---

## Current Projects

### 1. Daily Quote Sentiment Analysis

A beginner AI workflow that retrieves a random quote from an external API and analyzes its sentiment using a pretrained Hugging Face model.

#### Workflow

```text
DummyJSON Quotes API
        ↓
Random Quote and Author
        ↓
Hugging Face Sentiment Pipeline
        ↓
Sentiment Label and Confidence Score
```

#### Main Concepts

- REST API integration
- HTTP GET requests
- JSON response handling
- Hugging Face pipelines
- Pretrained sentiment models
- Text inference
- Error handling
- Reproducible notebooks

#### Project Location

```text
notebooks/daily_quote_sentiment/
```

---

### 2. AI Job Post Classifier — Model Selection Benchmark

An English zero-shot classification project for classifying technical job descriptions into career domains.

The project does not train a custom model from scratch. Instead, it follows a pretrained-first workflow:

```text
Define the Task
        ↓
Search Hugging Face Models
        ↓
Read Model Cards
        ↓
Shortlist Candidate Models
        ↓
Run Inference
        ↓
Build an Evaluation Dataset
        ↓
Compare Accuracy and Latency
        ↓
Select the Best Model
        ↓
Build a Gradio Demo
```

#### Job Categories

- Data Engineering and Data Pipelines
- Artificial Intelligence and Machine Learning
- DevOps and Deployment Automation
- Cloud Infrastructure and Cloud Engineering
- Cybersecurity and Information Security
- Software and Application Development

#### Candidate Models

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
3. `knowledgator/gliclass-small-v1.0-lw`

#### Comparison Criteria

- Classification accuracy
- Average inference latency
- Model size
- Memory requirements
- Ease of local execution
- Dependency complexity
- Ease of deployment
- Performance on ambiguous job descriptions

#### Current Status

Completed:

- Defined the project problem.
- Selected six career categories.
- Created descriptive candidate labels.
- Shortlisted three pretrained models.
- Downloaded and loaded `facebook/bart-large-mnli`.
- Successfully classified an initial Data Engineering job post.
- Displayed model scores using a Pandas DataFrame.
- Documented download and Windows cache issues.

In progress:

- Building the evaluation dataset.
- Testing the first model on all categories.
- Comparing the remaining candidate models.
- Measuring accuracy and latency.
- Selecting the final model.
- Building the Gradio demo.

#### Project Location

```text
notebooks/model_scout/
```

---

## Repository Structure

```text
ai-engineer-starter-kit/
│
├── notebooks/
│   │
│   ├── daily_quote_sentiment/
│   │   ├── daily_quote_sentiment.ipynb
│   │   └── README.md
│   │
│   └── model_scout/
│       ├── model_scout.ipynb
│       └── README.md
│
├── data/                       # Planned evaluation datasets
├── results/                    # Planned model comparison outputs
├── .gitignore
├── README.md
└── requirements.txt
```

The repository structure will expand as the roadmap progresses.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- Requests
- Hugging Face Transformers
- Hugging Face Hub
- PyTorch
- Pretrained NLP Models
- REST APIs
- Git
- GitHub

Planned:

- Scikit-learn
- Gradio
- Hugging Face Spaces

---

## Key Concepts Demonstrated

### AI and NLP

- Pretrained models
- Transformer-based models
- Inference pipelines
- Sentiment analysis
- Zero-shot classification
- Natural Language Inference
- Candidate-label design
- Confidence scores
- Model evaluation
- Accuracy and latency trade-offs

### Software and Engineering Practices

- Virtual environments
- Dependency management
- Modular project organization
- Error handling
- Git version control
- GitHub documentation
- Reproducible notebooks
- Technical decision documentation

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tarek-Radi/ai-engineer-starter-kit.git
cd ai-engineer-starter-kit
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 4. Install the Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open one of the project notebooks:

```text
notebooks/daily_quote_sentiment/daily_quote_sentiment.ipynb
```

or:

```text
notebooks/model_scout/model_scout.ipynb
```

In VS Code, select the Python kernel associated with the repository virtual environment:

```text
.venv
```

The interpreter path should look similar to:

```text
ai-engineer-starter-kit\.venv\Scripts\python.exe
```

---

## Reproducibility Checklist

Before committing a notebook:

1. Save the notebook.
2. Restart the kernel.
3. Run all cells from top to bottom.
4. Confirm that no hidden state is required.
5. Review the outputs.
6. Remove private information or tokens.
7. Update the project README.
8. Commit the working version to Git.

Recommended notebook test:

```text
Restart Kernel
→ Run All Cells
→ Verify Outputs
→ Save
→ Commit
```

---

## Important Development Lessons

Several practical issues were encountered during development:

### Hugging Face Model Downloads

Large model repositories may contain weights for multiple frameworks. Downloading an entire repository can use unnecessary disk space.

For BART, a selective download was used to retrieve only the required configuration, tokenizer, and SafeTensors files.

### Windows Hugging Face Cache

Windows may display symlink warnings when Developer Mode is disabled. The cache still works, but it may consume more disk space.

### Interrupted Downloads

Interrupted model downloads can leave incomplete files in the Hugging Face cache. These may need to be removed before retrying.

### Notebook Saving

Notebook changes must be saved before moving or renaming files. Unsaved cells may be lost because file operations only move the latest version stored on disk.

### Version Control

Frequent commits reduce the risk of losing notebook work and make it easier to restore earlier versions.

---

## Roadmap

### Completed

- External API integration
- JSON handling
- Pretrained sentiment inference
- Initial zero-shot model research
- Hugging Face model-card analysis
- First zero-shot job classification
- Basic Pandas result presentation

### In Progress

- Balanced evaluation dataset
- Multi-model comparison
- Accuracy measurement
- Latency benchmarking
- Reusable inference module
- Gradio demo

### Planned

- Hugging Face Spaces deployment
- More pretrained model tasks
- Image inference
- Audio inference
- Model-serving fundamentals
- RAG systems
- AI agents
- Evaluation and observability

---

## Expected Model Scout Deliverables

The Job Post Classifier project is expected to produce:

- A labeled evaluation dataset
- Three evaluated pretrained models
- Accuracy and latency measurements
- A model comparison table
- A clearly justified winning model
- A reusable inference module
- A Gradio web interface
- A public Hugging Face Space
- A final model selection report

---

## Author

**Tarek Mahmoud**

Engineering student interested in:

- Data Engineering
- Artificial Intelligence
- Cloud Computing
- AI-powered Data Systems
- Practical AI Engineering

GitHub: [Tarek-Radi](https://github.com/Tarek-Radi)

---

## License

This repository is currently intended for educational and portfolio purposes.

Individual pretrained models, datasets, APIs, and libraries used by the projects remain subject to their own licenses and terms of use.