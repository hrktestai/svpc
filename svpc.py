import streamlit as st
import pandas as pd
    


def intro():
    import streamlit as st

    st.write("# Customer's Potential Transaction Value Prediction ")

    st.sidebar.success("Select an Operations page.")

    st.markdown(f"### {list(page_names_to_funcs.keys())[0]}")

    st.markdown(
        """
        
        The prediction of customers' potential transaction value is a critical task in the field of marketing and customer relationship management. Accurate prediction of customers' transaction value can help businesses identify high-value customers, tailor marketing strategies, optimize resource allocation, and maximize revenue.
        
        In this study, we address the challenge of predicting customers' potential transaction value using machine learning techniques, specifically focusing on the Santander Value Prediction Challenge.The Santander Value Prediction Challenge is a Kaggle competition that provides a dataset containing a large number of anonymized features and the corresponding transaction value for each customer.
        
    """
    )
    
def page_1():

    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns

    
    st.markdown("## EDA on Train dataset  Demo")
    st.sidebar.header("Exploratory Data Analysis")
    st.write(
        """This demo in the page illustrates a heuristic view of the data provided to us
        I have build some plots to show the same ."""
    )
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    st.write("Train dataset rows and columns : ", train_df.shape)
    st.write("Test dataset rows and columns : ", test_df.shape)
    
    st.subheader("Train dataset (First 20 rows):")
    st.write(train_df.head(20))

    st.subheader("Test dataset (First 20 rows):")
    st.write(test_df.head(20))
    

    st.markdown(""" Observations :
    - We are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column
    - The train data and test data has 4992 unique Columns
    - The train data has 4459 rows
    - The test data has 49342 rows
    - In the train data , the number of columns is more than the number of train rows.
    - Test data has 10 times the samples as that of train set.
    """)

    
def page_2():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown("# EDA on Test dataset  Demo")
    st.sidebar.header("Plotting Demo")
    st.write(
        """### Plotting the Train dataset target values """
    )
  
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    

    
    plt.figure(figsize=(10,10))
    xrows = train_df.shape[0]
    ytval = train_df['target'].values
    plt.scatter(range(xrows), np.sort(ytval))
    plt.xlabel('index', fontsize=9)
    plt.ylabel('target', fontsize=9)
    plt.title("Target Distribution (Scatter Plot)", fontsize=9)
    st.pyplot(plt)

    plt.figure(figsize=(10,10))
    xrows = train_df.shape[0]
    ytval = train_df['target'].values
    plt.hist(np.sort(ytval), 50)
    plt.xlabel('index', fontsize=9)
    plt.ylabel('target', fontsize=9)
    plt.title("Target Distribution (Histogram) - by sorted target values", fontsize=9)
    st.pyplot(plt)

    st.write(
        """ Observations :
        The target variable distribution is highly skewed. """
    )
    
def trainedmodels():
 
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_squared_log_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np


    st.markdown("# Trained Models  Demo")
    st.sidebar.header("Trained Models Demo")
    st.write(
        """This demo in the page illustrates the performance of the models trained on the Train dataset ."""
    )



    # Load train_df and test_df data
    train_df = pd.read_csv("train.csv")

    # Extract features (X) and target (y) from train_df
    X = train_df.iloc[:, 2:]  # Extract all columns except the first column (ID)
    y = train_df.iloc[:, 1]  # Extract the second column as target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Load Linear Regression model from pickle file
    with open('lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    # Load Random Forest model from pickle file
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    # Load XGBoost model from pickle file
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
        
    # Load base_models.pkl
    with open('base_models.pkl', 'rb') as f:
        base_models = pickle.load(f)

    # Load meta_model.pkl
    with open('meta_model.pkl', 'rb') as f:
        meta_model = pickle.load(f)

    # Predict on test data
    y_pred_test_lr = lr_model.predict(X_test)
    y_pred_test_rf = rf_model.predict(X_test)
    y_pred_test_xgb = xgb_model.predict(X_test)

    #Predict using base models
    predictions_test = []
    for model in base_models:
        y_pred = model.predict(X_test)
        predictions_test.append(y_pred)

    #reate new dataset for meta model using base model predictions
    X_meta_test = np.column_stack(predictions_test)

    #Predict using meta model
    y_pred_meta_test = meta_model.predict(X_meta_test)

    # Print predicted values and their corresponding true expected values for all models
    st.write('Predicted vs Actual values on Train data:')
    st.write('---------------------------------------')

    # Create a DataFrame to store the predictions
    predictions_traindf = pd.DataFrame({'  LR Model': y_pred_test_lr,
                                    '  RF Model': y_pred_test_rf,
                                    '  XGB Model': y_pred_test_xgb,
                                    '  Custom Stacking Model': y_pred_meta_test,
                                    '  Actual Values (y_test)': y_test})

    # Set custom float format for displaying values
    pd.options.display.float_format = '{:,.2f}'.format

    st.title('Predictions DataFrame')
    
    # Display the DataFrame
    st.write(predictions_traindf.head(200))

    # Calculate RMSLE
    def rmsle(y_true, y_pred):
        """
        Calculate Root Mean Squared Logarithmic Error (RMSLE).

        Args:
            y_true (array-like): Array of true target values.
            y_pred (array-like): Array of predicted target values.

        Returns:
            float: RMSLE value.
        """
        assert len(y_true) == len(y_pred), "Input arrays must have the same length"
        y_true = np.log1p(y_true)
        y_pred = np.log1p(y_pred)
        squared_diff = (y_true - y_pred) ** 2
        mean_squared_error = np.mean(squared_diff)
        rmsle = np.sqrt(mean_squared_error)
        return rmsle

    st.title(' Root Mean Squared Logarithmic Error of the Model Predictions ')

    # Calculate LR RMSLE for test set
    rmsle_lr = rmsle(y_test, y_pred_test_lr)
    st.write("RMSLE Linear Regression Model: ", rmsle_lr)

    # Calculate RF RMSLE for test set
    rmsle_rf = rmsle(y_test, y_pred_test_rf)
    st.write("RMSLE RandomForest Model: ", rmsle_rf)

    # Calculate RMSLE for test set
    rmsle_xgb = rmsle(y_test, y_pred_test_xgb)
    st.write("RMSLE XGboost Model: ", rmsle_xgb)



    # Calculate RMSLE for test set
    rmsle_meta = rmsle(y_test, y_pred_meta_test)
    st.write("RMSLE Custom Stacking Model: ", rmsle_meta)



    #Scatter Plot of Custom Model Predictions vs Actual y_test values
    st.title('Actual vs Custom Model Prediction Values')
    
    # Plot y_test vs y_pred_meta_test
    plt.scatter(y_test, y_pred_meta_test)
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred_meta_test)')
    plt.title('Actual vs Predicted Values')
    
    # Display the plot using Streamlit
    st.pyplot(plt)



   
def test():
 
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_squared_log_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np


    st.markdown("# Test Dataset Prediction Demo")
    st.sidebar.header("Predictions on test.csv Demo")
    st.write(
        """This demo in the page illustrates the testdata predictions ."""
    )


    # Load test_df data
    test_df = pd.read_csv("test.csv")

    test_df = test_df.iloc[:, 1:]
    

    # Load Linear Regression model from pickle file
    with open('lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    # Load Random Forest model from pickle file
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    # Load XGBoost model from pickle file
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
        
    # Load base_models.pkl
    with open('base_models.pkl', 'rb') as f:
        base_models = pickle.load(f)

    # Load meta_model.pkl
    with open('meta_model.pkl', 'rb') as f:
        meta_model = pickle.load(f)

    # Predict on test data
    ydf_pred_test_lr = lr_model.predict(test_df)
    ydf_pred_test_rf = rf_model.predict(test_df)
    ydf_pred_test_xgb = xgb_model.predict(test_df)

    #Predict using base models
    predictions_test = []
    for model in base_models:
        y_pred = model.predict(test_df)
        predictions_test.append(y_pred)

    #reate new dataset for meta model using base model predictions
    X_meta_test = np.column_stack(predictions_test)

    #Predict using meta model
    ydf_pred_meta_test = meta_model.predict(X_meta_test)

    # Print predicted values and their corresponding true expected values for all models
    st.write('Predicted values on Test data:')
    st.write('---------------------------------------')

    # Create a DataFrame to store the predictions
    predictions_testdf = pd.DataFrame({'  LR Model': ydf_pred_test_lr,
                                    '  RF Model': ydf_pred_test_rf,
                                    '  XGB Model': ydf_pred_test_xgb,
                                    '  Custom Stacking Model': ydf_pred_meta_test
                                    })

    # Set custom float format for displaying values
    pd.options.display.float_format = '{:,.2f}'.format

    st.title('Testdata Predictions DataFrame')
    
    # Display the DataFrame
    st.write(predictions_testdf.head(50000))

  


page_names_to_funcs = {
    "Introduction": intro,
    "EDA on Datasets ": page_1,
    "EDA on Target Distribution": page_2,
    "Trained Models": trainedmodels,
    "Model Prediction": test,
}

demo_name = st.sidebar.selectbox("Choose an Operation", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
