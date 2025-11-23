# MinneMUDAC-2025

**Big Brothers Big Sisters Mentorship Program Analysis**

A comprehensive data analytics and machine learning pipeline for predicting mentorship match outcomes and identifying risk factors in the Big Brothers Big Sisters of America program.

<img width="2127" height="1196" alt="image" src="https://github.com/user-attachments/assets/3176427a-6375-410c-9d3f-f9b8db42af8a" />

## 🚀 Key Innovation: LLM-Based Semantic Analysis

### **Grok Prompt Pipeline: Advanced Semantic Understanding**

This project leverages **Large Language Models (LLMs)** for semantic text analysis, representing a significant advancement over traditional NLP approaches.

#### **Why LLM-Based Analysis?**

Unlike traditional NLP methods that rely on:
- Keyword matching and pattern recognition
- Rule-based classification systems
- Limited context understanding
- Pre-defined feature extraction

Our **Grok Prompt Pipeline** uses **semantic understanding** to:
- **Understand context and nuance** in unstructured text notes
- **Identify implicit events** that may not be explicitly stated
- **Recognize relationships** between different concepts and events
- **Adapt to domain-specific terminology** (BBB, MEC, MSC, etc.)
- **Extract structured information** from free-form text with high accuracy

#### **How It Works**

The Grok Prompt Pipeline processes mentorship support contact notes using the Grok LLM to:

1. **Semantic Event Detection**: Identifies 35+ different event types (match closures, family moves, volunteer issues, etc.) by understanding the semantic meaning of text, not just keywords.

2. **Green Flag & Red Flag Classification**: Uses contextual understanding to classify events as positive indicators (green flags) or risk factors (red flags) that may lead to early match termination.

3. **Severity Scoring**: Assigns severity scores (1-5) to detected events based on semantic analysis of the text's tone, context, and implications.

4. **Conversation Context Management**: Maintains conversation history per match ID, allowing the LLM to understand temporal context and relationship evolution over time.

#### **Traditional NLP vs. LLM Approach**

| Traditional NLP | LLM-Based (Grok) |
|----------------|------------------|
| Keyword matching | Semantic understanding |
| Rule-based patterns | Context-aware analysis |
| Limited to explicit mentions | Captures implicit meaning |
| Requires extensive feature engineering | Learns from examples |
| Brittle to variations | Robust to language variations |
| Cannot handle domain-specific nuances | Understands domain context |

#### **Example**

**Input Text:**
> "MEC noted that the volunteer expressed some concerns about time constraints due to new work schedule. However, both parties remain committed to meeting regularly."

**Traditional NLP:** Might miss the nuanced concern or classify it incorrectly.

**LLM Analysis:** Understands this as:
- Event: "Volunteer: Time constraints" (severity: 2)
- Green Flag: Continued commitment despite challenges
- Context: Potential risk mitigated by commitment

---

## 📁 Project Structure

```
MinneMUDAC-2025/
├── Grok Prompt Pipeline/          # ⭐ LLM-based semantic analysis
│   └── grok_prompt_processor.py  # Main pipeline with real-time validation
│
├── ML Pipeline/                   # Advanced ML training
│   └── advanced_ml_training.py    # Ensemble models + LSTM
│
├── MUDAC/                        # Data and notebooks
│   ├── External Data/            # Raw datasets
│   ├── Prompt/                   # Grok prompt notebooks
│   └── Completed/                # Processed datasets
│
└── BI/                           # Business Intelligence dashboards
    └── PowerBI dashboards
```

---

## 🔧 Components

### 1. **Grok Prompt Pipeline** ⭐

**Location:** `Grok Prompt Pipeline/grok_prompt_processor.py`

**Purpose:** Extract structured event data from unstructured mentorship notes using LLM semantic analysis.

**Key Features:**
- Real-time validation and error checking
- Conversation context management per match
- JSON response parsing and validation
- Event name validation against 35+ predefined categories
- Severity score validation (1-5 range)
- Comprehensive error logging

**Output:**
- Structured event columns (35+ event types)
- Green flag and red flag counts
- Severity scores for each detected event
- Processed notes with semantic insights

**Usage:**
```python
from grok_prompt_processor import main

main(
    input_file="MUDAC/External Data/Test-Truncated-Restated.xlsx",
    output_file="Grok Prompt Pipeline/output_df.xlsx"
)
```

### 2. **Advanced ML Training Pipeline**

**Location:** `ML Pipeline/advanced_ml_training.py`

**Purpose:** Predict match length using ensemble methods and LSTM time series models.

**Key Features:**
- **Advanced Feature Engineering:**
  - Time-based features (cyclical encoding, temporal patterns)
  - Interaction features (multiplications, divisions, differences)
  - Aggregation features (mean, std, min, max per match)
  - Polynomial features
  
- **Ensemble Methods:**
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Stacked ensemble with meta-learner

- **Advanced LSTM Architecture:**
  - Bidirectional LSTM
  - Attention mechanism
  - Residual connections
  - Early stopping and gradient clipping

**Usage:**
```python
from advanced_ml_training import main

main()  # Uses config defaults
```

---

## 🎯 Workflow

### Step 1: Semantic Analysis (Grok Prompt Pipeline)
1. Load mentorship support contact notes
2. Process text through Grok LLM for semantic understanding
3. Extract events, flags, and severity scores
4. Validate and structure output

### Step 2: Feature Engineering & ML Training
1. Combine LLM-extracted features with structured data
2. Apply advanced feature engineering
3. Train ensemble models (RF, GB, XGBoost)
4. Train LSTM for time series patterns
5. Create stacked ensemble for final predictions

### Step 3: Analysis & Visualization
1. PowerBI dashboards for insights
2. Event pattern analysis
3. Risk factor identification
4. Match outcome predictions

---

## 📊 Data Sources

- **Mentorship Match Data**: Match IDs, dates, participant information
- **Support Contact Notes**: Unstructured text notes (processed by Grok)
- **Demographic Data**: Age, location, income, education
- **Event Data**: Structured events extracted via LLM semantic analysis
- **Census Data**: Minnesota block group demographics

---

## 🛠️ Dependencies

### Grok Prompt Pipeline
- `pandas` - Data manipulation
- `openai` - Grok API client
- `nltk` - Text processing
- `numpy` - Numerical operations

### ML Pipeline
- `pandas`, `numpy` - Data processing
- `torch` - Deep learning (LSTM)
- `scikit-learn` - Traditional ML models
- `xgboost` - Gradient boosting
- `scipy` - Statistical operations

---

## 🔑 Key Features

### ✅ LLM-Based Semantic Analysis
- Understands context and nuance
- Extracts implicit information
- Handles domain-specific terminology
- Robust to language variations

### ✅ Real-Time Validation
- Input validation at every step
- JSON structure validation
- Event name validation
- Error logging and recovery

### ✅ Advanced ML Pipeline
- Multiple model architectures
- Ensemble methods
- Time series modeling
- Comprehensive feature engineering

### ✅ Production-Ready
- Modular design
- Error handling
- Model persistence
- Comprehensive logging

<img width="2041" height="1160" alt="image" src="https://github.com/user-attachments/assets/8d9d88e4-a463-4bdd-abb6-d8fb25798911" />
<img width="2045" height="1156" alt="image" src="https://github.com/user-attachments/assets/a024b3c6-4581-4833-92bd-bd1d32ee5ddc" />
<img width="1912" height="1068" alt="image" src="https://github.com/user-attachments/assets/dd1ff124-df48-4a3a-93b7-851733a11752" />

---

## 📈 Results & Insights

The LLM-based approach enables:
- **Higher accuracy** in event detection compared to keyword-based methods
- **Better understanding** of implicit risks and positive indicators
- **Reduced false positives** through semantic context
- **Scalability** to new event types without retraining

---

## 🏆 Competition Results

### MinneMUDAC 2025 - Round #1 Results

**Team:** U37  
**Overall Score:** 60.29 / 80

#### Individual Rubric Scores

| Rubric Item | Score |
|------------|-------|
| **Creativity & Innovation** | **3.50** |
| Communication of Outcomes & Team Synergy | 3.10 |
| Prediction | 3.03 |
| Impact of Important Factors | 3.00 |
| Completeness & Breadth of Outcome | 2.80 |
| Appropriateness of Analytical Methods | 2.80 |

#### Judge Feedback

> "LLM Calling is an interesting idea - but consider cost of implementations. Slides were too busy. No survey questions."

#### Highlights

- **Highest score in Creativity & Innovation (3.50)** - Validates the innovative LLM-based semantic analysis approach
- **Strong performance in Prediction (3.03)** - Demonstrates effectiveness of the ensemble ML pipeline
- **Above-average Communication score (3.10)** - Effective presentation of complex technical concepts

The **3.50 score in Creativity & Innovation** recognizes the project's novel use of LLM-based semantic analysis, which represents a significant advancement over traditional NLP methods for extracting insights from unstructured mentorship notes.

#### Competition Context

- **Mean Overall Score (All Teams):** 57.2
- **Our Score:** 60.29 (above average)
- **Score Range:** 43.9 - 71.7
- **Our Percentile:** ~75th percentile

---

## 🚧 Future Enhancements Retro

- Fine-tune LLM on domain-specific data
- Multi-model ensemble for LLM responses
- Real-time processing pipeline
- Integration with match management system

---

## 📝 Notes

- The Grok Prompt Pipeline is the **core innovation** of this project, demonstrating the power of LLM-based semantic analysis over traditional NLP.
- All models and preprocessors are saved for reproducibility.
- Error logs are maintained for debugging and improvement.

---

## 👥 Team

Minnesota Undergraduate Data Analytics Competition 2025

---

## 📄 License

This project is part of the MinneMUDAC 2025 competition.
