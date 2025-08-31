# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning educational repository containing implementations of various ML algorithms and techniques from "Machine Learning A-Z" course. The repository is structured as a learning progression from basic data preprocessing through advanced topics like deep learning.

## Architecture and Structure

### Dual Implementation Structure
The repository follows a dual-track approach:
- **Part 1 - Data Processing/**: Custom learning implementations with Chinese comments for educational clarity
- **datasets/Machine-Learning-A-Z-Codes-Datasets/**: Complete official course materials with standardized structure

### Course Progression (Official Materials)
Located in `datasets/Machine-Learning-A-Z-Codes-Datasets/`:
- **Part 1**: Data Preprocessing (missing data, encoding, scaling)
- **Part 2**: Regression (Linear, Polynomial, SVR, Decision Tree, Random Forest)
- **Part 3**: Classification (Logistic, KNN, SVM, Naive Bayes, Decision Tree, Random Forest)
- **Part 4**: Clustering (K-Means, Hierarchical)
- **Part 5**: Association Rule Learning (Apriori, Eclat)
- **Part 6**: Reinforcement Learning (UCB, Thompson Sampling)
- **Part 7**: Natural Language Processing
- **Part 8**: Deep Learning (ANN, CNN)
- **Part 9**: Dimensionality Reduction (PCA, LDA, Kernel PCA)
- **Part 10**: Model Selection and Boosting (Grid Search, XGBoost)

### Section Structure Pattern
Each algorithm section follows this hierarchy:
```
Section X - Algorithm Name/
├── Python/
│   ├── algorithm_name.py          # Main implementation
│   ├── algorithm_name.ipynb       # Jupyter notebook version
│   └── dataset.csv                # Algorithm-specific data
└── R/
    ├── algorithm_name.R
    └── dataset.csv
```

## Running Code

### Python Execution
Navigate to the specific section directory:
```bash
cd "datasets/Machine-Learning-A-Z-Codes-Datasets/Part X - Topic/Section Y - Algorithm/Python"
python algorithm_name.py
```

### Jupyter Notebooks
All sections include both `.py` and `.ipynb` versions:
```bash
jupyter notebook algorithm_name.ipynb
```

### Custom Learning Files
For personal implementations in "Part 1 - Data Processing/":
```bash
cd "Part 1 - Data Processing"
python script_name.py
```

## Development Patterns

### Standard Data Preprocessing Template
All implementations follow this sequence (see `data_preprocessing_tools.py`):
```python
# 1. Data Import
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# 2. Handle Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 3. Encode Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 4. Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 5. Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
```

### Dataset Path Conventions
- **Official course files**: CSV in same directory as Python/R scripts
- **Custom implementations**: Reference `datasets/Data.csv` or `datasets/titanic.csv`
- **Always verify paths** when moving between sections

### Visualization Standards
Consistent plotting patterns across algorithms:
- Training data: Red scatter plots (`color='red'`)
- Model predictions: Blue lines (`color='blue'`)
- Chinese titles in custom implementations for learning clarity
- Standard English titles in official course materials

### Code Documentation Style
- **Official course files**: Minimal comments, focus on implementation
- **Custom learning files**: Extensive Chinese comments explaining each step
- **Mixed approach**: English variable names with Chinese explanatory comments

## Key Libraries and Dependencies

Core ML stack used throughout:
- `pandas`: Data manipulation and CSV handling
- `numpy`: Numerical computations and array operations  
- `matplotlib.pyplot`: Basic plotting and visualization
- `scikit-learn`: All ML algorithms, preprocessing, and model selection
- Additional libraries per section: `seaborn`, `tensorflow`, `xgboost`