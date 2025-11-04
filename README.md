# FYP-2025-26 ML Cryoprotectant

## Guide of Machine Learning Workflow for Cryopreservation Optimization

### **1. Introduction and Overview**

#### **1.1 Project Goal**
This project aims to optimize DMSO-free cryopreservation formulations using machine learning algorithms. You will:
- Clean and format literature data from Excel spreadsheets
- Use differential evolution to find optimal formulations
- Train three different ML models: Random Forest, Gradient Boosting, and Neural Networks
- Compare predictions and select the best candidates for experimental validation

#### **1.2 What is Machine Learning?**
Machine learning (ML) algorithms learn patterns from existing data to make predictions about new situations. In this project:
- **Input data**: Existing cryopreservation formulations (DMSO concentration, cooling rate, protective proteins, sugars, etc.)
- **Output**: Cell viability predictions
- **Goal**: Find formulations that maximize viability while minimizing DMSO

#### **1.3 Workflow Overview**
The complete workflow follows these steps:
1.	**Literature Data Collection** → Excel spreadsheet
2.	**Data Cleaning & Formatting** → Python/Pandas processing
3.	**Machine Learning Models** → Differential Evolution, Random Forest, Gradient Boosting, Neural Network
4.	**Optimal Formulation Predictions** → Top candidates identified
5.	**Laboratory Validation** → Experimental testing

### **2. Software Installation and Setup**
#### **2.1 Install Anaconda & Python**
1.Sign up using CityU's email address and download [Anaconda](https://www.anaconda.com) for your operating system (Windows/Mac/Linux)
1. Choose ``Python 3.13`` if prompted
1. Run the installer with default settings
1. Verify installation:
Open "Anaconda Prompt" (Windows) or Terminal (Mac/Linux)
Type: ``python --version``
You should see: ``Python 3.13``

#### **2.2 Install Required Libraries**
Open Anaconda Prompt (or Command Prompt/Terminal) and type these commands **one by one**:

    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install scipy
    pip install tensorflow
    pip install matplotlib
    pip install openpyxl
 
Wait for each installation to complete before moving to the next. You should see ``"Successfully installed..."`` messages.

#### **2.3 Create a GitHub account, and forking**
If you don't already have a [GitHub](https://github.com) account, sign up for one using your most frequently used email address, then [fork the current files](https://github.com/DoggoneBastard/FYP-2025-26.git) into your own repository.

#### **2.4 Code Editor**
We are going to use [Visual Studio Code](https://code.visualstudio.com/). Download and install it.

After you open the app, go to ``File - Open...``, then  select a **local directory**, rather than on iCloud or OneDrive, and open.

Press ``Ctrl+Shift+P`` (Windows/Linux) or ``Cmd+Shift+P`` (macOS). Then type ``Python: Select Interpreter``, then find the one you previously installed using Anaconda. It should have ``Conda`` displayed on the right side.

Now you need to link your GitHub with VS Code. To do this, follow [this guide](https://code.visualstudio.com/docs/sourcecontrol/github), and clone your repository that you have forked to the local directory you selected previously. All your code files and results will be saved here; this will be the directory in which you work in.

### **3. Data Preparation and Formatting**
#### **3.1 Why Data Preparation is Critical**
Raw data from literature is messy and inconsistent:
- Viability reported as "82.5%" or "0.825" or "82.5 ± 3.2%"
- Cooling rates as "1°C/min" or "-1/min to -80°C" or "Directly at -80˚C"
- Missing values in many columns
- Inconsistent units and formats
**Machine learning algorithms require clean numeric data in a consistent format**. This is why data preprocessing typically takes **70-80%** of the time in any ML project.

#### **3.2 Understanding Your Excel Data**
The Excel file currently has two sheets:
- **MSC**: Mesenchymal stem cells data (569 rows)
- **Melanocyte**: Primary human melanocytes data (6 rows)

Key columns:
- **All ingredients in cryoprotective solution**: Text description of formulation
- **DMSO usage**: Concentration (0 to 1 or percentage)
- **Cooling rate**: Temperature change rate (°C/min)
- **Viability**: Cell survival post-thaw (0-1 or percentage)
- **Protective protein type**: e.g., HSA, BSA, FBS
- **Non-reducing sugar type**: e.g., trehalose, sucrose
- **PEG weight**: Molecular weight of polyethylene glycol
- **Alternative CPA type**: Alternative cryoprotective agents

#### **3.3 Data Cleaning Strategy**
Transformations needed:
1.	**Viability**: Extract numeric value, convert percentages to decimals (0-1 range)
2.	**Cooling rate**: Extract °C/min value from text
3.	**DMSO**: Ensure numeric format (0-1 range)
4.	**Missing values**: Strategically fill or remove

Example transformations:
> "82.5 ± 3.2%" → 0.825

> "91.4 ± 2.7%" → 0.914

> "1°C/min" → 1.0

> "-1/min to -80°C" → 1.0 
    >> (keep signs consistent)

> "Directly at -80˚C" -> ? 
    >> Rate refer to the separate Excel file in GitHub

#### **3.4 Step-by-Step: Data Preparation Code**
**Read file**: ``Step1_Data_Preparation.py``

**What this script does**:
1.	Loads Excel data using pandas library
2.	Applies text processing to extract numeric values
3.	Handles missing values (fills with median or zero)
4.	Saves cleaned data as CSV file

**Key functions**:

    def extract_cooling_rate(cooling_str):
        """Extract primary cooling rate from cooling rate string"""
        # Looks for patterns like "1°C/min", "-1/min"
        # Returns numeric value

    def extract_viability(viability_str):
        """Extract numeric viability value from text"""
        # Handles percentages, decimals, ± notation
        # Returns value between 0 and 1
**To run**:
1.	Open VS Code
1.	Run the script

**Expected output**:

    Cleaned data saved! Shape: (353, 21)

This means you successfully cleaned 353 valid data entries with 21 columns, and the file is saved to the same local directory.

#### **3.5 Verify Your Cleaned Data**
Open ``Cleaned_Cryopreservation_Data.csv`` in Excel or a text editor. Check to see if there's any obvious data or labeling errors. Your job will be to go back to the sample code and add more variables as we go along. Work smarter using chatbots.

### **4. Differential Evolution Optimization**
#### **4.1 What is Differential Evolution?**
Differential Evolution (DE) is an evolutionary optimization algorithm inspired by natural selection:
1.	**Initialize**: Create a population of candidate solutions (formulations)
2.	**Evaluate**: Test how good each solution is (predict viability)
3.	**Mutate**: Create new solutions by combining existing ones
4.	**Select**: Keep the better solutions, discard worse ones
5.	**Repeat**: Continue until convergence
**Analogy**: Imagine breeding plants. You start with 15 different varieties, test which grow best, crosspollinate the best ones to create new varieties, and repeat. Eventually, you find the optimal variety.

#### **4.2 How DE Works for Cryopreservation**
**Problem**: We want to find the formulation (DMSO concentration, cooling rate) that maximizes cell viability.

**Challenge**: We can't test thousands of formulations in the lab—it would take years...

**Solution**: Use a surrogate model:
1.	Train a Random Forest model on existing literature data
2.	Use this model to quickly predict viability for ANY formulation
3.	DE uses these fast predictions to explore the search space
4.	After finding the optimum with DE, validate the top candidates in the lab

#### **4.3 Key DE Parameters**
- **Population size (popsize)**: Number of candidate solutions (default: 15)
- **Maximum iterations (maxiter)**: Number of generations (default: 100)
- **Mutation factor (F)**: Controls mutation strength (default: 0.5-1.0)
- **Crossover probability (CR)**: Controls mixing of solutions (default: 0.7-0.9)
- **Bounds**: Search space limits; For example
    - **DMSO**: 0% to 10% (focusing on DMSO-free or low-DMSO)

#### **4.4 Step-by-Step: Differential Evolution**
**Read file**: ``Step2_Differential_Evolution.py``

**What this script does**:
1.	Loads cleaned data
2.	Splits into training and testing sets
3.	Trains a Random Forest surrogate model
4.	Defines the optimization objective (maximize viability)
5.	Runs differential evolution
6.	Reports the optimal formulation

**Key code structure**:

    # Define objective function
    def objective_function(params):
        # params[0] = DMSO concentration
        # params[1] = Cooling rate
        predicted_viability = surrogate_model.predict([params])[0]
        return -predicted_viability  # Minimize negative = maximize positive

    # Define search bounds
    bounds = [(0, 0.15)]      # DMSO: 0% to 15%

    # Run optimization result = differential_evolution(objective_function, bounds, ...)


**Expected output**:

    Surrogate model R² score: 0.XXX
    ============================================================
    DIFFERENTIAL EVOLUTION RESULTS
    ============================================================
    Optimal DMSO concentration: 0.0234 (2.34%)
    Optimal cooling rate: 1.23 °C/min
    Predicted maximum viability: 0.8567 (85.67%)
    ============================================================
    Results saved to: DE_Optimization_Results.csv

#### **4.5 Interpreting DE Results**
**R² score**: Measures how well the surrogate model fits the data

1.0 = Perfect fit  
0.8-0.9 = Very good  
0.6-0.8 = Good  
0.6 = Poor (may need more data or better features)

But in our cases R² of more than 0.20 is acceptable.

**Optimal formulation**: The DMSO concentration and cooling rate predicted to give the highest viability

**Predicted viability**: Expected cell survival—but remember, this is a prediction only. You must validate experimentally.

#### **4.6 When to Use DE**
**Advantages**:
- Doesn't require gradient information
- Handles non-linear, multi-modal objective functions
- Relatively simple to implement
- Few hyperparameters to tune
**Limitations**:
- Computationally expensive for high-dimensional problems
- No guarantee of finding global optimum
- Relies on surrogate model accuracy
**Best for**: Finding optimal values of 2-5 continuous parameters (like DMSO concentration and maybe cooling rate).

### **5. Machine Learning Models**
#### **5.1 Overview of Three Algorithms**

We'll train three different types of models and compare their predictions:

**Random Forest Regressor**
- **How it works**: Creates many decision trees, each trained on a random subset of data. Final prediction is the average of all trees.
- **Strengths**: Handles non-linear relationships, robust to outliers, provides feature importance
- **Weaknesses**: Can be slow with very large datasets, may overfit with too many trees
- **Best for**: Interpretable models, understanding which features matter most

**Gradient Boosting Regressor**
- **How it works**: Builds trees sequentially, where each new tree corrects the errors of previous trees
- **Strengths**: Often very accurate, good at capturing complex patterns
- **Weaknesses**: More prone to overfitting, requires careful hyperparameter tuning
- **Best for**: Maximizing prediction accuracy, especially with tabular data

**Neural Network (TensorFlow/Keras)**
- **How it works**: Multiple layers of interconnected nodes (neurons) that learn to transform inputs into outputs through backpropagation
- **Strengths**: Can learn highly complex, non-linear relationships
- **Weaknesses**: Needs more data, computationally intensive, "black box" (hard to interpret)
- **Best for**: Very complex patterns, large datasets

#### **5.2 Step-by-Step: Training All Models**
**Read file**: ``Step3_ML_Models.py``

**What this script does**:
1.	Loads cleaned data
2.	Splits into training (80%) and testing (20%) sets
3.	Trains Random Forest, Gradient Boosting, and Neural Network
4.	Evaluates each model on test data
5.	Uses each model to find optimal formulation
6.	Compares all models in a summary table

**To run**:
1.	Ensure you completed Step 3.4 (data cleaning)
2.	Open Step3_ML_Models.py in VS Code
3.	Run the entire script
4.	Wait 1-3 minutes for training to complete

**Expected output**:

    ============================================================
    RANDOM FOREST REGRESSOR
    ============================================================
    R² Score: 0.XXXX
    Mean Absolute Error: 0.XXXX
    Root Mean Squared Error: 0.XXXX
    Feature Importance:
    DMSO Concentration: 0.XXXX
    Cooling Rate: 0.XXXX
    Optimal formulation (Random Forest):
    DMSO: X.XX%
    Cooling Rate: X.XX °C/min
    Predicted Viability: XX.XX%

#### **5.3 Understanding Evaluation Metrics**

**R² Score**: See above; Higher the bettter.

**Mean Absolute Error (MAE)**
- Average absolute difference between predictions and actual values
- Lower is better
- Example: MAE = 0.05 means predictions are off by ±5% on average
- Easy to interpret: same units as target variable

**Root Mean Squared Error (RMSE)**
- Square root of average squared errors
- Lower is better
- Penalizes large errors more than MAE
- More sensitive to outliers

**Which metric to use?**  
**R²**: Overall model quality  
**MAE**: Average error magnitude  
**RMSE**: When large errors are particularly bad

#### **5.4 Feature Importance (Random Forest)**
Random Forest provides **feature importance scores** that show which variables most influence predictions:

    Feature Importance:
        DMSO Concentration: 0.6234
        Cooling Rate: 0.3766

**Interpretation**: DMSO concentration has ~62% influence on viability, while cooling rate has ~38% influence.

**Why this matters**:
- Helps you understand what drives cell survival
- Guides experimental focus
- Identifies which parameters to optimize first

#### **5.5 Model Comparison Summary**
After running all models, you'll see a comparison table like this:

    ============================================================
    MODEL COMPARISON SUMMARY
    ============================================================
        Model         R² Score MAE    RMSE      Optimal DMSO (%)  Optimal Cooling (°C/min) 
    Random Forest       0.XXX  0.XXX  0.XXX     X.XX                      X.XX     
    Gradient Boosting   0.XXX  0.XXX  0.XXX     X.XX                      X.XX     
    Neural Network      0.XXX  0.XXX  0.XXX     X.XX                      X.XX     

How to choose the best model:
1.	**Highest R²** → Best overall fit to the data
2.	**Lowest MAE/RMSE** → Most accurate predictions
3.	**Consistency** → Do all models predict similar optimal formulations? If yes, more confident. If no, more uncertainty.

#### **5.6 Advanced: Hyperparameter Tuning**
The provided code uses default hyperparameters. For better performance, you can tune:

**Random Forest:**
- ``n_estimators``: Number of trees (try 50, 100, 200, 500)
- ``max_depth``: Maximum tree depth (try 5, 10, 20, None)
- ``min_samples_split``: Minimum samples to split node (try 2, 5, 10)

**Gradient Boosting**:
- ``n_estimators``: Number of boosting stages (try 50, 100, 200)
- ``learning_rate``: Step size shrinkage (try 0.01, 0.1, 0.3)
- ``max_depth``: Tree depth (try 3, 5, 7)

**Neural Network**:
- Number of layers (try 2, 3, 4 hidden layers)
- Neurons per layer (try 16, 32, 64, 128)
- Dropout rate (try 0.1, 0.2, 0.3)
- Learning rate (usually handled by Adam optimizer)

**How to tune**: See scikit-learn documentation for tutorials.

### **6. Results Interpretation and Experimental Design**
#### **6.1 Synthesizing Predictions from Multiple Models**

You now have predictions from four methods:
1.	Differential Evolution (using Random Forest surrogate)
2.	Random Forest direct optimization
3.	Gradient Boosting direct optimization
4.	Neural Network direct optimization

**Key analysis questions**:
- **Agreement**: Do all methods predict similar optimal formulations?
    - High agreement → More confidence in predictions
    - Low agreement → More uncertainty, test multiple candidates
- **Viability predictions**: Are predicted viabilities realistic?
    - 60-95% is typical for cryopreservation
    - \>95% may indicate model overfitting
    - < 50% suggests poor formulation
- **DMSO levels**: How much DMSO is recommended?
    - 0-5%: Excellent for DMSO-free goal
    - 5-10%: Good reduction from typical 10%
    - \> 10%: May need to reconsider bounds or add more features

#### **6.2 Expanding the Feature Space**
Currently the models only use 2 features (DMSO, cooling rate). To include more ingredients:

**Additional features to consider**:
1.	**Protective proteins**: HSA, BSA, FBS (type and concentration)
2.	**Non-reducing sugars**: Trehalose, sucrose (type and concentration)
3.	**Polymers**: PEG (molecular weight and concentration), HES
4.	**Other CPAs**: Glycerol, ethylene glycol, propylene glycol
5.	**Basal medium**: DMEM, α-MEM, PBS
6.	**Freezing parameters**: Nucleation temperature, hold time, warming rate

**Data engineering steps**:
1.	Extract concentrations from "All ingredients" text column using pattern matching
2.	One-hot encode categorical variables (e.g., protein type: HSA=1, BSA=0, FBS=0)
3.	Normalize concentrations to consistent units (M, mg/mL, %)
4.	Handle missing values more carefully (e.g., 0 for absent ingredient)

**Example enhanced feature set**:
- DMSO_Numeric
- Cooling_Rate_Numeric
- Trehalose_Concentration
- HSA_Concentration
- PEG_MW (molecular weight)
- Has_FBS (binary: 0 or 1)

**Trade-off**: More features can capture more complex relationships but require more training data and increase risk of overfitting.

### **7. References and Further Learning**
#### **7.1 Python Programming and Data Science**
**Online Courses**:
- Coursera: "Python for Everybody" (University of Michigan)
- DataCamp: "Introduction to Python for Data Science"
- Kaggle Learn: Free micro-courses on Python, Pandas, ML

**Documentation/Additional Help Resources**:
1.	Stack Overflow: Search or ask coding questions (https://stackoverflow.com/)
2.	Python documentation: Official docs (https://docs.python.org/)
3.	Scikit-learn docs: ML library documentation (https://scikit-learn.org/)
4.	TensorFlow tutorials: Neural network guides (https://www.tensorflow.org/tutorials)
1. Pandas: https://pandas.pydata.org/docs/
1. NumPy: https://numpy.org/doc/
1. Matplotlib: https://matplotlib.org/stable/contents.html
5.	YouTube tutorials
6.	Chatbots like ChatGPT/Claude: Ask AI assistants for code help

#### **7.2 Debugging strategy**
1.	Read error message carefully: It usually tells you exactly what's wrong
2.	Google the error: Someone has likely encountered it before
3.	Print intermediate values: Use print() statements to check data at each step
4.	Simplify: Comment out complex parts, test simple version first
5.	Ask for help: Supervisor, classmates, online forums

#### **7.3 Machine Learning Introductory**
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Google's Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course
- [StatQuest YouTube Channel](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw): Excellent visual explanations

**For Specific Algorithms**:
- Random Forest: https://www.datacamp.com/tutorial/random-forests-classifier-python
- Gradient Boosting: https://www.machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
- Neural Networks: https://www.tensorflow.org/tutorials/quickstart/beginner

**Advanced Topics**:
Hyperparameter tuning: ``GridSearchCV``, ``RandomizedSearchCV``
Cross-validation: ``k-fold``, ``leave-one-out``
Feature engineering: Creating informative features
Ensemble methods: Combining multiple models

#### **7.4 Optimization Algorithms**
**Differential Evolution**
- SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
- Tutorial: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

**Other Optimization Methods**:
- **Bayesian Optimization**: For expensive objective functions
- **Genetic Algorithms**: Discrete optimization problems

### **8. Best Practices and Recommendations**
#### **8.1 Data Management**
**Organization**:
- Keep raw data unchanged in a separate folder
- Version your scripts (e.g., `script_v1.py`, `script_v2.py`)
- Document changes in a lab notebook or `README` file
- Use meaningful variable names (`viability_numeric` not `x`)

**Reproducibility**:
- Set random seeds for reproducible results:

        import numpy as np
        import random

        np.random.seed(42)
        random.seed(42)
- Save trained models for future use:

        import joblib

        joblib.dump(rf_ model,'random_forest_model.pkl')
        # Load later: rf_model = joblib.load('random_forest_model.pkl)

#### **8.2 Code Quality**
**Comments**:
- Explain **why**, not just *what*
- Document function inputs and outputs

        def extract_viability(viability_str):

            """
            Function:
                Extract numeric viability value from text
            Args:
                viability_str: String containing viability (e.g., "82.5%", "0.825")
            Returns:
                Float between 0 and 1, or NaN if not extractable
            """

**Testing**:
- Test functions on sample inputs before applying to full dataset
- Verify outputs make sense (e.g., viability should be 0-1)

#### **8.3 Model Development**
**Train-Test Split**:
- Always hold out test data (20-30%)
- Never use test data for training or tuning
- Test data simulates "unseen" real-world data

**Cross-Validation**:
- Use k-fold cross-validation (k=5 or 10) for more robust evaluation
- Helps detect overfitting

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"CV R² scores: {scores}")
        print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

**Feature Engineering**:
- Start simple (few features)
- Add complexity gradually
- Validate that new features improve performance

**Avoid Overfitting**:
- Use simpler models when data is limited
- Apply regularization (e.g., Ridge, Lasso regression)
- Prune decision trees (limit depth, min samples per leaf)
- Use dropout in neural networks

### **9.Final Advice**
1.	Start simple: Master basic models before advanced techniques
1.	Be careful, document everything, and have back-ups: Future you will thank present you
1.	Be patient: ML has a learning curve, but it's worth it

### **Good luck with the project!**
--- ---------------------
Appendix A: Key Terminology  
Algorithm: Step-by-step procedure for solving a problem or making calculations  
Backpropagation: Method for training neural networks by propagating errors backward through layers  
Bias-Variance Tradeoff: Balance between model simplicity (high bias) and complexity (high variance)  
Cross-Validation: Technique for assessing model performance by testing on multiple data subsets  
Feature: Input variable or predictor used by ML model (e.g., DMSO concentration)  
Feature Engineering: Creating new features from existing data to improve model performance  
Hyperparameter: Model configuration setting that controls learning (e.g., number of trees in Random Forest)  
Overfitting: When model memorizes training data but fails to generalize to new data  
Regularization: Techniques to prevent overfitting by penalizing model complexity  
Surrogate Model: Approximate model used in place of expensive objective function  
Target (or Label): Output variable being predicted (e.g., viability)  
Training: Process of model learning patterns from data  
Validation: Assessing model performance on data not used for training  

Appendix B: Sample Code Snippets
Loading and Exploring Data

    import pandas as pd
    # Load Excel file
    df = pd.read
    _
    excel('Cryopreservative-Data-Oct.27.xlsx'
    , sheet
    name=
    _
    'MSC')
    # Basic exploration
    print(df.shape) # (rows, columns)
    print(df.columns) # Column names
    print(df.head()) # First 5 rows
    print(df.describe()) # Summary statistics
    print(df['Viability'].value
    _
    counts()) # Count unique values

Handling Missing Values

    # Check for missing values
    print(df.isnull().sum())
    # Fill missing values with median
    df['Cooling_
    Rate
    _
    Numeric'].fillna(df['Cooling_
    Rate
    _
    Numeric'].median(), inplace=True)
    # Fill missing values with zero
    df['DMSO
    _
    Numeric'].fillna(0, inplace=True)
    # Drop rows with any missing values
    df
    _
    clean = df.dropna()

Train-Test Split

    from sklearn.model_selection import train_test_split
    X = df[['DMSO_Numeric','Cooling_Rate_Numeric']].values
    y = df['Viability_Numeric'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, randomstate=42
    )
 
Cross-Validation 

    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"CV R² scores: {scores}")
    print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

--- - 
<center> <b> End of Guide </b> </center>