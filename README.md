# 🚗 Interactive Actuarial P&C Pricing Simulator

> A web-based pricing tool for personal auto insurance using industry-standard GLM methodology

## 🎯 Project Overview

This project demonstrates a complete actuarial pricing workflow, from raw data processing to an interactive web application. It uses the French Motor Third Party Liability (MTPL2) dataset to build a frequency-severity GLM model that calculates pure premiums for auto insurance policies.

**Key Features:**
- ✅ Industry-standard Poisson-Gamma GLM modeling
- ✅ Interactive web interface built with Streamlit
- ✅ Real-time premium calculation and factor analysis
- ✅ Waterfall charts showing premium decomposition
- ✅ Sensitivity analysis visualizations

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Modeling** | Python (statsmodels, scikit-learn) | GLM fitting and predictions |
| **Data Processing** | Pandas, NumPy | Data manipulation and feature engineering |
| **Visualization** | Plotly, Matplotlib | Charts and model diagnostics |
| **Deployment** | Streamlit Cloud | Zero-friction web deployment |

## 📊 Actuarial Methodology

### Two-Part GLM Model

**Frequency Model (Poisson GLM):**
- **Target:** Number of claims per policy
- **Distribution:** Poisson with log link
- **Offset:** Exposure (policy years)
- **Features:** Driver age, vehicle characteristics, region, etc.

**Severity Model (Gamma GLM):**
- **Target:** Average claim amount (for policies with claims)
- **Distribution:** Gamma with log link
- **Features:** Same rating factors as frequency model

**Pure Premium = Predicted Frequency × Predicted Severity**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/actuarial-pricing-simulator.git
cd actuarial-pricing-simulator
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Go to [Kaggle French Motor Insurance Dataset](https://www.kaggle.com/datasets/florianbordesoule/french-motor-claim-datasets-fremtpl2-and-fremtpl2sev)
   - Download `freMTPL2freq.csv` and `freMTPL2sev.csv`
   - Place files in the `data/` directory

5. **Run the data pipeline:**
```bash
# Step 1: Data exploration
python 01_data_exploration.py

# Step 2: Feature engineering
python 02_feature_engineering.py

# Step 3: GLM modeling
python 03_glm_modeling.py
```

6. **Launch the web app:**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
actuarial-pricing-simulator/
├── data/                          # Data files and processed datasets
│   ├── freMTPL2freq.csv          # Raw frequency data
│   ├── freMTPL2sev.csv           # Raw severity data
│   ├── frequency_processed.parquet
│   ├── severity_processed.parquet
│   └── trained_models.pkl
├── utils/                         # Utility functions
├── 01_data_exploration.py         # Initial EDA
├── 02_feature_engineering.py      # Feature preparation
├── 03_glm_modeling.py            # GLM training
├── app.py                        # Streamlit web interface
├── modeling.py                   # Model utilities
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🎨 Web Interface Features

### 1. **Interactive Input Panel**
- Driver age, vehicle characteristics, location
- Real-time premium updates as you change inputs
- Intuitive sliders and dropdown menus

### 2. **Premium Breakdown**
- Waterfall chart showing how each factor impacts the final premium
- Base premium + adjustments = final premium
- Clear visualization of risk factor contributions

### 3. **Sensitivity Analysis**
- Charts showing how premium varies with key factors
- Price elasticity analysis
- Interactive plots with Plotly

### 4. **Model Insights**
- Coefficient tables with statistical significance
- Model performance metrics
- Diagnostic plots (expandable sections)

## 📈 Model Performance

The GLM models achieve industry-standard performance:

- **Frequency Model:** Poisson deviance with proper exposure handling
- **Severity Model:** Gamma GLM capturing right-skewed claim distributions
- **Validation:** Out-of-sample testing with MAE and RMSE metrics

## 🚀 Live Demo

[**🔗 Try the Live App**](https://your-app-url.streamlit.app)

## 📸 Screenshots

### Premium Calculator Interface
![Premium Calculator](screenshots/premium_calculator.png)

### Waterfall Chart
![Waterfall Chart](screenshots/waterfall_chart.png)

### Sensitivity Analysis
![Sensitivity Analysis](screenshots/sensitivity_analysis.png)

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- **Actuarial Science:** GLM methodology, frequency-severity modeling, rating factor analysis
- **Data Science:** Feature engineering, statistical modeling, model validation
- **Software Engineering:** Clean code structure, version control, documentation
- **Product Development:** User interface design, interactive visualizations, deployment

## 🔄 Development Workflow

### Week 1: Data & Modeling Foundation
- [x] Data acquisition and exploration
- [x] Feature engineering and binning
- [x] GLM model fitting and validation

### Week 2: Web Interface Development
- [ ] Streamlit app structure
- [ ] User input forms and prediction backend
- [ ] Interactive visualizations

### Week 3: Polish & Deployment
- [ ] UI/UX improvements
- [ ] Code refactoring and documentation
- [ ] Streamlit Cloud deployment

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset:** French Motor Third Party Liability dataset from Kaggle
- **Methodology:** Based on industry-standard actuarial pricing practices
- **Libraries:** Built with the amazing Python data science ecosystem

## 📞 Contact

**Rayyan Merchant** - [rayyanmerch@gmail.com](mailto:rayyanmerch@gmail.com)

Project Link: [https://github.com/yourusername/actuarial-pricing-simulator](https://github.com/yourusername/actuarial-pricing-simulator)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
