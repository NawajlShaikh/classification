import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

class Util:
    def __init__(self, file_path='data/file.csv'):
        self.features = ['age', 'sex', 'albumin', 'alkaline_phosphatase', 'alanine_aminotransferase',
                         'aspartate_aminotransferase', 'bilirubin', 'cholinesterase', 'cholesterol',
                         'creatini','gamma_glutamyl_transferase', 'protein']
        
        self.target_col ='Diagnosis'
        self.file_path = file_path
        #self.df = self.get_data()  # Get the data during initialization

        # Check if the target column exists in the DataFrame
        #if self.target_col not in self.df.columns:
        #   raise ValueError(f"'{self.target_col}' column not found in the DataFrame.")

        # Check class distribution before preprocessing
       # class_distribution_before = self.df[self.target_col].value_counts()
       # if len(class_distribution_before) < 2:
       #     raise ValueError("Insufficient samples for both classes before preprocessing. Check class distribution.")
        
        # Preprocess the data
       # self.df = self.preprocess(self.df)

        # Check class distribution after preprocessing
        #class_distribution = self.df[self.target_col].value_counts()
       # if len(class_distribution) < 2:
            # If one class is underrepresented, oversample or handle class imbalance appropriately
         #   raise ValueError("Insufficient samples for both classes. Check class distribution.")


    def preprocess(self, df):
        # Rename columns for diagnosis classes
        df.rename(columns={'category': 'Diagnosis'}, inplace=True)
        df[self.target_col] = df[self.target_col].apply(lambda x:1 if x==0 else 0)
        # Convert categorical to numerical column
        df['sex'] = df['sex'].apply(lambda x:1 if x=='m' else 0) 
        return df
    
    @st.cache(allow_output_mutation=True)
    
    def get_data(self):
        df = pd.read_csv(self.file_path)
        # preprocess data
        df = self.preprocess(df)
        return df

    @st.cache
    def split_data(self, df):
        X = df[self.features]
        y = df[self.target_col]

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print("Class distribution after SMOTE:")
        print(y_resampled.value_counts())


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
        return X_train, X_test, y_train, y_test

    @st.cache(allow_output_mutation=True)
    def build_model(self, X, y):
        model = xgb.XGBClassifier()
        print("Fitting the model")
        model.fit(X, y)
        return model


    def compute_accuracy(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred) * 100

    def predict(model, X):
        prediction = model.predict(X)
        return prediction

    def input_data_fields(self, overwrite_vals=None):
        default_vals = {'age': 32, 'sex': 1, 'albumin': 38.5, 'alkaline_phosphatase': 52.5, 'alanine_aminotransferase': 7.7,
                        'aspartate_aminotransferase': 22.1, 'bilirubin': 7.5, 'cholinesterase': 6.93, 'cholesterol': 3.23,
                        'creatini':106 ,'gamma_glutamyl_transferase': 12.1, 'protein': 69 }

        col1, col2 = st.columns(2)
        age = col1.number_input("age",
                            min_value=None,
                            step=5,
                            value=default_vals['age'],
                            help="In the United States, the average age at onset of liver cancer is 63 years.")

        sex = col2.selectbox('sex',
                            ('Male', 'Female'),
                            index=default_vals['sex'],
                            help="Men are more likely to develop liver cancer than women, by a ratio of 2 to 1.")

        albumin = col1.number_input("albumin (G/dL)",
                            min_value=0.0,  # Set a specific minimum value based on your requirements
                            step=0.5,       # Use a step that matches the type of min_value and value
                            value=default_vals['albumin'],
                            help="The normal range is 38.5 to 40.1 g/dL (34 to 54 g/L).")

        alkaline_phosphatase = col1.number_input("alkaline_phosphatase (IU/L)", 
                            min_value=None,
                            value=default_vals['alkaline_phosphatase'],
                            help="The normal range is 44 to 147 international units per liter (IU/L).")

        bilirubin = col1.number_input("bilirubin (mg/dL)", 
                            min_value=None,
    
                            value=default_vals['bilirubin'], 
                            help="It is normal to have some bilirubin in the blood. A normal level is: 0.1 to 1.2 mg/dL (1.71 to 20.5 µmol/L)")
        
        cholinesterase = col2.number_input("cholinesterase (mg/dL)", 
                            min_value=None,
                 
                            value=default_vals['cholinesterase'], 
                            help="Normal level for Direct (also called conjugated) cholinesterase is less than 0.3 mg/dL.")
        
        alanine_aminotransferase = col2.number_input("alanine_aminotransferase (U/L)", 
                            min_value=None,         
                            value=default_vals['alanine_aminotransferase'],
                            help="The normal range is 4 to 36 U/L.")
        
        aspartate_aminotransferase = col1.number_input("aspartate_aminotransferase (U/L)", 
                            min_value=None,
                            value=default_vals['aspartate_aminotransferase'],
                            help="The normal range is 8 to 33 U/L.")
        
        cholesterol = col2.number_input("cholesterol (mg/dL)", 
                            min_value=None,
                            step=0.5,
                            value=default_vals['cholinesterase'], 
                            help="Normal level for Direct (also called conjugated) cholstrol is less than 0.3 mg/dL.")
        
        creatini = col1.number_input("creatini (mg/dL)",
                            min_value=0.0,  # Set a specific minimum value based on your requirements
                            step=0.1,
                            value=float(default_vals['creatini']),
                            help="Normal level for Direct (also called conjugated) creatini is less than 0.3 mg/dL.")

                    
        gamma_glutamyl_transferase = col2.number_input("gamma_glutamyl_transferase", 
                            min_value=None,
                            step=0.2,
                            value=default_vals['gamma_glutamyl_transferase'],
                            help="The normal range for albumin/globulin ratio is over 1 , usually around 1 to 2.")
      
        protein = col2.number_input("protein (g/dL)", 
                            min_value=None,
                            value=default_vals['protein'],
                            help="The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")

        sex = 0 if sex == "m" else 1

        return {'age': age,
                'sex': sex,
                'albumin': albumin,
                'alkaline_phosphatase': alkaline_phosphatase,
                'alanine_aminotransferase': alanine_aminotransferase,
                'aspartate_aminotransferase': aspartate_aminotransferase,
                'bilirubin': bilirubin,
                'cholinesterase': cholinesterase,
                'cholesterol': cholesterol,
                'creatini': creatini,
                'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
                'protein': protein }
    

    def form_functions(self, model):
        with st.form("my_form"):
            get_values = self.input_data_fields()
            # Add the submit button
            submitted = st.form_submit_button("Submit", type="primary")
        
        if submitted:
            data_values = pd.DataFrame([get_values])

            # Get predictions
            with st.spinner('Making prediction...'):
                time.sleep(3)

            prediction = model.predict(data_values)
            print("Prediction: ", prediction[0])

            # Define class labels and corresponding messages
            class_labels = ["No Disease", "Suspect disease", "Hepatitis", "Fibrosis", "Cirrhosis"]
            disease_name = class_labels[prediction[0]]
            prediction_msg = f"The supplied values suggest that the patient has {disease_name}."

            st.subheader("Diagnosis:")

            if prediction[0] == 0:
                st.success(prediction_msg)
            elif prediction[0] in [1, 2, 3, 4]:
                st.warning(prediction_msg)
            else:
                st.error(prediction_msg)


    def sample_data(self, df):
        test_data = df.drop('Diagnosis', axis=1).to_dict(orient='records')
        return test_data

    def page_footer(self):
        footer = """<style>
            a:link , a:visited{
            color: blue;
            background-color: transparent;
            text-decoration: underline;
            }

            a:hover,  a:active {
            color: red;
            background-color: transparent;
            text-decoration: underline;
            }

            .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            color: white;
            text-align: center;
            }
            </style><div class="footer"><p>Developed by <a style='display: block; text-align: center;' target="_blank">Group No.3  Final Presentation </a></p></div>
            """

        return footer
