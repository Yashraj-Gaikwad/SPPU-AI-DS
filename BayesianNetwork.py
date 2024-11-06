'''
Write a program to construct a Bayesian network considering medical data. Use this model to
demonstrate the diagnosis of heart patients using the standard Heart Disease Data Set (You can use
Java/Python ML library classes/API.

Dataset Description
    age
    sex
    cp (chest pain type)
    trestbps (resting blood pressure)
    chol (serum cholesterol)
    fbs (fasting blood sugar)
    restecg (resting electrocardiographic results)
    thalach (maximum heart rate achieved)
    exang (exercise induced angina)
    oldpeak (depression induced by exercise relative to rest)
    slope (slope of the peak exercise ST segment)
    ca (number of major vessels colored by fluoroscopy)
    thal (thalassemia)
    target (diagnosis of heart disease)


viva questions
1) Explain bayesian network

'''

# Import Libraries
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

# Load the dataset
heart_disease = pd.read_csv('heart.csv')

# Check for missing values and data types
print(heart_disease.info())
print(heart_disease.isnull().sum())

# Sample a smaller dataset for testing purposes
heart_disease_sample = heart_disease.sample(n=100, random_state=42)

# Discretize continuous variables (example for 'chol' and 'trestbps')
heart_disease_sample['chol'] = pd.cut(heart_disease_sample['chol'], bins=[0, 200, 240, 300], labels=[0, 1, 2])
heart_disease_sample['trestbps'] = pd.cut(heart_disease_sample['trestbps'], bins=[0, 120, 140, 180], labels=[0, 1, 2])

# Check unique values after discretization
print("Unique values for 'chol':", heart_disease_sample['chol'].unique())
print("Unique values for 'trestbps':", heart_disease_sample['trestbps'].unique())

# Constructing a simplified Bayesian Network structure
model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('chol', 'target'),
    ('trestbps', 'target')
])

# Learning CPDs using Maximum Likelihood Estimators on sampled data
model.fit(heart_disease_sample, estimator=MaximumLikelihoodEstimator)

# Performing inference with Bayesian Network
infer = VariableElimination(model)

# Example Queries:
query_result = infer.query(variables=['target'], evidence={'age': 52, 'sex': 1, 'cp': 0})
print('\nProbability of Heart Disease given age=52, sex=1, cp=0:')
print(query_result)

# Use valid state names based on discretization
query_result2 = infer.query(variables=['target'], evidence={'trestbps': 1, 'chol': 1}) # Adjusted to valid states
print('\nProbability of Heart Disease given trestbps=130 (1), chol=250 (1):')
print(query_result2)
