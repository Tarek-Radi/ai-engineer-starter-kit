# AI Engineer Starter Kit

A beginner-friendly AI project that retrieves a random quote from an external API and analyzes its sentiment using a Hugging Face pretrained model.

This project demonstrates how to combine external data sources with an AI model inside a clean and reproducible Jupyter Notebook.

---

## Project Overview

The project retrieves a random quote from the DummyJSON Quotes API.

The quote is then passed to a Hugging Face sentiment analysis pipeline to determine whether the text expresses a positive or negative sentiment.

The project workflow is:

```text
External API
    ↓
Random Quote
    ↓
Hugging Face Sentiment Analysis
    ↓
Sentiment Label and Confidence Score
```

---

## Project Objectives

The main objectives of this project are to:

- Call an external REST API using Python.
- Work with JSON responses.
- Extract specific values from API data.
- Use a pretrained Hugging Face model.
- Perform AI inference on text.
- Display the result clearly.
- Build a clean and reproducible AI project.
- Document the project professionally on GitHub.

---

## Technologies Used

- Python
- Jupyter Notebook
- Requests
- Hugging Face Transformers
- PyTorch
- DummyJSON Quotes API
- Git
- GitHub

---

## External API

This project uses the DummyJSON Quotes API to retrieve a random quote.

API endpoint:

```text
https://dummyjson.com/quotes/random
```

The API returns a JSON response similar to:

```json
{
  "id": 1,
  "quote": "Your heart is the size of an ocean.",
  "author": "Rumi"
}
```

The project extracts the following values:

```python
quote_text = data["quote"]
author_name = data["author"]
```

---

## Hugging Face Pipeline

The project uses the Hugging Face Transformers library.

The sentiment analysis pipeline is created using:

```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
```

The quote is then passed to the model:

```python
result = sentiment_analyzer(quote_text)
```

The model returns a result similar to:

```python
[
    {
        "label": "POSITIVE",
        "score": 0.9987
    }
]
```

The label represents the predicted sentiment, while the score represents the confidence of the model.

---

## Project Structure

```text
ai-engineer-starter-kit/
│
├── notebooks/
│   └── daily_quote_sentiment.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

### File Description

- `README.md`: Contains the project documentation.
- `requirements.txt`: Contains the required Python libraries.
- `.gitignore`: Prevents unnecessary files from being uploaded.
- `notebooks/daily_quote_sentiment.ipynb`: Contains the full project implementation.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tarek-Radi/ai-engineer-starter-kit.git
```

Move into the project directory:

```bash
cd ai-engineer-starter-kit
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install the Required Libraries

```bash
python -m pip install -r requirements.txt
```

---

## How to Run the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/daily_quote_sentiment.ipynb
```

Then run all cells from top to bottom.

In VS Code, make sure that the selected Notebook Kernel uses the project virtual environment:

```text
.venv
```

The Python executable should point to a path similar to:

```text
ai-engineer-starter-kit\.venv\Scripts\python.exe
```

---

## Example Output

```text
Quote:
The only limit to our realization of tomorrow is our doubts of today.

Author:
Franklin D. Roosevelt

Sentiment:
POSITIVE

Confidence:
99.87%
```

---

## Main Code Workflow

The API request is sent using:

```python
response = requests.get(url, timeout=10)
```

The JSON response is converted into a Python dictionary:

```python
data = response.json()
```

The quote and author are extracted:

```python
quote_text = data["quote"]
author_name = data["author"]
```

The quote is analyzed using the Hugging Face model:

```python
result = sentiment_analyzer(quote_text)
```

The final result is displayed using:

```python
print(f"Quote: {quote_text}")
print(f"Author: {author_name}")
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2%}")
```

---

## Error Handling

The API request includes a timeout to prevent the notebook from waiting indefinitely:

```python
response = requests.get(url, timeout=10)
```

The project also checks whether the API request was successful:

```python
response.raise_for_status()
```

This raises an error when the API returns an unsuccessful status code.

Examples include:

- `404`: The requested resource was not found.
- `500`: The API server experienced an internal error.
- `Timeout`: The server did not respond within the specified time.
- `ConnectionError`: The application could not connect to the API.

---

## Reproducibility

The notebook is designed to run from top to bottom without errors.

Before submission, the notebook should be tested using:

```text
Restart Kernel
Clear All Outputs
Run All
```

The project does not require a private API key.

All required dependencies are included in:

```text
requirements.txt
```

---

## Key Concepts Demonstrated

This project demonstrates several important AI engineering concepts:

- REST API integration
- HTTP GET requests
- JSON data handling
- Text preprocessing
- Pretrained models
- Hugging Face pipelines
- AI inference
- Virtual environments
- Dependency management
- GitHub documentation
- Reproducible notebooks

---

## Future Improvements

Possible future improvements include:

- Analyzing multiple quotes.
- Saving the results to a CSV file.
- Adding neutral sentiment classification.
- Creating a Streamlit user interface.
- Comparing multiple sentiment analysis models.
- Adding charts for sentiment results.
- Deploying the project as a web application.

---

## Author

**Tarek Mahmoud**

Engineering student interested in:

- Data Engineering
- Artificial Intelligence
- Cloud Computing
- AI-powered Data Systems

GitHub: [Tarek-Radi](https://github.com/Tarek-Radi)

---

## License

This project is created for educational purposes.