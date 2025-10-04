# Student Performance Indicator (SPI) - Machine Learning Project

## 📊 Project Overview

The **Student Performance Indicator (SPI)** is an end-to-end machine learning project that predicts student math scores based on various demographic and academic factors. This project demonstrates the complete ML pipeline from data ingestion to model deployment using Flask web framework.

## 🎯 Objective

The primary objective is to build a machine learning model that can predict a student's math score based on:
- **Demographic factors**: Gender, race/ethnicity
- **Educational background**: Parental level of education, test preparation course completion
- **Academic performance**: Reading and writing scores
- **Socioeconomic factors**: Lunch type (free/reduced vs standard)

## 🏆 Results

The project successfully:
- ✅ Predicts math scores with high accuracy using multiple ML algorithms
- ✅ Provides an interactive web interface for real-time predictions
- ✅ Implements a complete MLOps pipeline with proper data preprocessing
- ✅ Uses ensemble methods and hyperparameter tuning for optimal performance

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **Flask** - Web framework for API and UI
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization

### Machine Learning Libraries
- **XGBoost** - Gradient boosting framework
- **CatBoost** - Gradient boosting with categorical features
- **Scikit-learn** - Traditional ML algorithms (Random Forest, SVM, etc.)

### Development Tools
- **Jupyter Notebook** - Data exploration and model development
- **Docker** - Containerization
- **HTML/CSS/JavaScript** - Frontend interface

## 📁 Project Structure

```
ml_spi-main/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── setup.py                       # Package configuration
├── Dockerfile                     # Docker configuration
├── README.md                      # Project documentation
├── artifacts/                     # Model artifacts
│   ├── model.pickle              # Trained model
│   ├── preprocessor.pickle      # Data preprocessor
│   ├── data.csv                 # Raw dataset
│   ├── train.csv                # Training data
│   └── test.csv                 # Testing data
├── src/                          # Source code
│   ├── components/              # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── config/                  # Configuration files
│   ├── pipeline/                # Training and prediction pipelines
│   ├── exception.py             # Custom exception handling
│   ├── logger.py               # Logging configuration
│   └── utils.py                # Utility functions
├── notebook/                    # Jupyter notebooks
│   ├── EDA_SPI.ipynb           # Exploratory Data Analysis
│   ├── MODEL_TRAINING.ipynb    # Model training notebook
│   └── data/
│       └── Stud data.csv       # Original dataset
└── templates/                   # HTML templates
    ├── index.html              # Homepage
    └── home.html               # Prediction form
```

## 🚀 Installation Steps

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ml_spi-main
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

### 4. Verify Installation
```bash
python -c "import flask, sklearn, pandas, numpy; print('All dependencies installed successfully!')"
```

## 🏃‍♂️ How to Run

### Method 1: Direct Flask Application
```bash
# Navigate to project directory
cd ml_spi-main

# Run the Flask application
python app.py
```

The application will start on `http://localhost:80` (or `http://127.0.0.1:80`)

### Method 2: Using Docker (Recommended)
```bash
# Build Docker image
docker build -t ml-spi-app .

# Run Docker container
docker run -p 80:80 ml-spi-app
```

### Method 3: Development Mode
```bash
# Set Flask environment
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development

# Run with debug mode
python app.py
```

## 🌐 Usage

### Web Interface
1. **Open your browser** and navigate to `http://localhost:80`
2. **Click "Start Prediction"** to access the prediction form
3. **Fill in the required information**:
   - Gender (Male/Female)
   - Race/Ethnicity (Group A-E)
   - Parental Education Level
   - Lunch Type (Free/Reduced or Standard)
   - Test Preparation Course (None/Completed)
   - Reading Score (0-100)
   - Writing Score (0-100)
4. **Click "Predict Math Score"** to get the prediction
5. **View the predicted math score** displayed on the same page

### API Usage
The application also provides a REST API endpoint:
```bash
POST /predictdata
Content-Type: application/x-www-form-urlencoded

# Form data:
gender=male&race_ethnicity=group_A&parental_level_of_education=bachelor's_degree&lunch=standard&test_preparation_course=completed&reading_score=85&writing_score=90
```

## 📊 Dataset Information

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size**: 1000 rows × 8 columns
- **Features**:
  - `gender`: Student's gender
  - `race_ethnicity`: Student's race/ethnicity group
  - `parental_level_of_education`: Parent's education level
  - `lunch`: Type of lunch (free/reduced or standard)
  - `test_preparation_course`: Whether student completed test prep
  - `reading_score`: Reading test score (0-100)
  - `writing_score`: Writing test score (0-100)
  - `math_score`: Math test score (0-100) - **Target Variable**

## 🔧 Model Training

### Training Process
1. **Data Preprocessing**: Handle categorical variables, scale numerical features
2. **Feature Engineering**: Create meaningful features from raw data
3. **Model Selection**: Test multiple algorithms (Random Forest, XGBoost, CatBoost, etc.)
4. **Hyperparameter Tuning**: Optimize model parameters using cross-validation
5. **Model Evaluation**: Assess performance using R² score, RMSE, and MAE

### Model Performance
- **Best Algorithm**: XGBoost/CatBoost (ensemble methods)
- **Evaluation Metrics**: R² Score, RMSE, MAE
- **Cross-validation**: 5-fold CV for robust evaluation

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t student-performance-predictor .

# Run the container
docker run -p 80:80 student-performance-predictor
```

### Docker Compose (if available)
```yaml
version: '3.8'
services:
  ml-spi-app:
    build: .
    ports:
      - "80:80"
    environment:
      - FLASK_ENV=production
```

## 🧪 Testing

### Run Tests
```bash
# Run unit tests (if available)
python -m pytest tests/

# Test the web interface
curl -X POST http://localhost:80/predictdata \
  -d "gender=male&race_ethnicity=group_A&parental_level_of_education=bachelor's_degree&lunch=standard&test_preparation_course=completed&reading_score=85&writing_score=90"
```

## 📈 Performance Monitoring

### Model Metrics
- **R² Score**: Measures explained variance
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold CV for robust evaluation

### Logging
The application includes comprehensive logging for:
- Model predictions
- Error handling
- Performance metrics
- User interactions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Krupali Bhagat**
- Email: krupalibhagat1155@gmail.com
- GitHub: [@krupali-bhagat](https://github.com/krupali1105)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Flask community for excellent documentation
- Scikit-learn team for comprehensive ML tools
- Open source community for various libraries and tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the author at krupalibhagat1155@gmail.com

---

**Happy Predicting! 🎓📊**