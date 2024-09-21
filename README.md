# Churn Prediction Dashboard


The **Churn Prediction Dashboard** is a web-based application that predicts customer churn based on input features such as tenure, monthly charges, contract type, and payment method. The app is built using **Streamlit**, and a machine learning model (Random Forest Classifier) provides real-time predictions for whether a customer will churn. Interactive visualizations using **Plotly** make it easy for users to interpret the prediction results.

This project not only demonstrates proficiency in machine learning but also shows skills in handling **imbalanced datasets** (via SMOTE) and deploying a user-friendly web app. The data used in this project is sourced from the **Telco Customer Churn Dataset** on Kaggle.


## Key Features

-   **Real-Time Churn Prediction**: Users can input customer details and receive immediate predictions on whether the customer is likely to churn.
-   **Interactive Visualization**: The dashboard includes dynamic bar charts to visualize churn probabilities.
-   **SMOTE for Imbalanced Data**: The model uses **Synthetic Minority Over-sampling Technique (SMOTE)** to handle the imbalanced dataset and improve predictions for minority classes (churned customers).
-   **Random Forest Model**: A **Random Forest Classifier** is used for predicting churn, leveraging its ability to handle both numerical and categorical features effectively.
-   **Data Preprocessing**: The project involves extensive data preprocessing, including handling missing values, encoding categorical features, and scaling numerical features.


## Model Building Process

### 1. **Data Preprocessing**:

-   Convert categorical variables like `gender` and `Contract` into numerical form using **`pd.get_dummies()`**.
-   Handle missing values in the `TotalCharges` column by converting it to numeric and filling missing values with the median.
-   Scale numerical features such as `tenure`, `MonthlyCharges`, and `TotalCharges` using **`StandardScaler`** for better model performance.

### 2. **Handling Imbalanced Data**:

-   **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to balance the dataset. Churn is a rare event, so oversampling the minority class (churned customers) ensures the model is not biased toward non-churn predictions.

### 3. **Model Training**:

-   The processed data is split into **training (80%)** and **testing (20%)** sets.
-   A **Random Forest Classifier** is trained on the oversampled data to predict customer churn.

### 4. **Evaluation**:

-   The model's performance is evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. An accuracy of **88%** was achieved on the test set.

## Dataset

The dataset used for training the machine learning model is the **Telco Customer Churn Dataset**, sourced from Kaggle. This dataset includes information about customer demographics, services subscribed, and account information such as tenure and monthly charges.

**Dataset Summary**:

-   **Total Rows**: 7,043
-   **Columns**: 21 features
-   **Key Features**:
    -   `gender`, `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, etc.
-   **Target Variable**: `Churn` (Yes/No)


## How the Project Works

1.  **User Input**: Users provide key customer details such as tenure, gender, and contract type through a sidebar form.
2.  **Churn Prediction**: The input data is passed through a **Random Forest Classifier** to predict whether the customer will churn.
3.  **Probability Visualization**: The app displays the probabilities for both churn and no churn using a **Plotly bar chart**.
4.  **Instant Results**: The prediction updates in real-time as user inputs change, providing an interactive and dynamic experience.


## Installation Instructions

### Prerequisites

-   **Python 3.7+** installed on your system.
-   **pip** for installing dependencies.

### Steps

1.  **Clone the repository**:
    
   
    
    `git clone https://github.com/laibashakil/churn-prediction-dashboard.git` 
    
2.  **Navigate into the project directory**:
    
   
    
    `cd churn-prediction-dashboard` 
    
3.  **Create a virtual environment (optional but recommended)**:
    

    
    `python -m venv env` 
    
4.  **Activate the virtual environment**:
    
    -   On Windows:
        

        
        `.\env\Scripts\activate` 
        
    -   On macOS/Linux:
        
        
        `source env/bin/activate` 
        
5.  **Install the required dependencies**:
   
    
    `pip install -r requirements.txt` 
    
6.  **Run the application**:
    
    `streamlit run app.py` 
    
7.  **Open your browser** and go to the URL provided by Streamlit (typically `http://localhost:8501`) to view and interact with the app.
    

## Technologies Used

-   **Python**: Core programming language for model training and building the web app.
-   **Streamlit**: Provides the framework for the interactive web dashboard.
-   **scikit-learn**: Used for building and training the **Random Forest Classifier** and handling data preprocessing.
-   **pandas**: For manipulating and preprocessing data.
-   **Plotly**: For creating dynamic and interactive data visualizations in the app.
-   **imbalanced-learn (SMOTE)**: To handle class imbalance in the dataset and improve the prediction of the minority class (churned customers).


## Model Details

The churn prediction model is a **Random Forest Classifier** trained using the **Telco Customer Churn Dataset**. This classifier is particularly suited for classification tasks like churn prediction due to its ability to handle both categorical and numerical data effectively.

Key model parameters:

-   **n_estimators**: 100
-   **max_depth**: 10
-   **random_state**: 42

The model was evaluated using accuracy, precision, recall, and F1-score. The overall accuracy on the test set was around **88%**.


## Screenshots
![WhatsApp Image 2024-09-21 at 21 38 28_b054e32e](https://github.com/user-attachments/assets/2828b623-f9d2-4d4d-8f33-dea3583a7dd4)
![WhatsApp Image 2024-09-21 at 21 46 57_ee088c55](https://github.com/user-attachments/assets/a9c3d0f4-0b10-4063-8d42-78861f2e92e8)


## Future Enhancements

-   **Cloud Deployment**: Deploy the app on **Heroku** or **AWS** for real-time customer churn predictions accessible from anywhere.
-   **Feature Importance Analysis**: Add more advanced visualizations such as **feature importance** or **decision tree paths** to provide deeper insights into which factors influence customer churn the most.
-   **Integration with CRMs**: Connect the app with customer relationship management (CRM) platforms for real-time business use.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.
