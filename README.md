# Text Emotion  Classification
![image](https://github.com/user-attachments/assets/b094432f-e9e2-44a4-a944-9805b04ce943)


### This project classifies text into one of four emotions: Joy, Sadness, Anger, or Fear. The system uses a Recurrent Neural Network (RNN) model for high accuracy and integrates with a Streamlit application for real-time emotion prediction.

## Project Overview

### 1. Data Preparation

- **Data Collection:**
  - The dataset used is collected from Kaggle. You can find it [here](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

- **Data Cleaning and Filtering:**
  - Removed duplicate rows.
  
- **Text Preprocessing:**
  - Cleaned texts by removing non-alphabetic characters, converting to lowercase, removing stopwords, and applying stemming.

- **Feature Extraction:**
  - Trained a Word2Vec model to convert text into vector representations.
  - Computed average word vectors for each text sample.

- **Dataset Splitting:**
  - Split data into training and validation sets.
  - Encoded labels for classification.

### 2. Model Training

- **Gradient Boosting Models:**
  - Used Optuna for hyperparameter tuning of `LGBMClassifier` and `XGBClassifier`.
  - Evaluated cross-validation scores to select the best model.

- **RNN Model:**
  - Built a Sequential model with an Embedding layer, SimpleRNN layer, and Dense output layer.
  - Achieved an accuracy of approximately **89%** with this RNN model, which significantly outperforms traditional machine learning models that achieve **61-64%** accuracy.

- **Model and Tokenizer Saving:**
  - Saved the trained RNN model, tokenizer, and label encoder for future use.

### 3. Streamlit Application

A Streamlit application is developed to allow users to input text and receive emotion predictions in real-time. For more details on running the Streamlit application, refer to the following instructions:

1. **Install Dependencies:**
   - Ensure you have the required libraries installed:
     ```bash
     pip install streamlit pandas numpy pickle5 nltk keras tensorflow
     ```

2. **Run the Streamlit App:**
   - Navigate to the directory containing your Streamlit script and run:
     ```bash
     streamlit run your_script_name.py
     ```

3. **Interact with the App:**
   - Open the provided URL in your browser to use the text classification app.

## Additional Notes

- Ensure that the `maxlen` parameter in the Streamlit app matches the value used during model training.
- The model and tokenizer files (`emotion_classification_rnn.h5`, `tokenizer.pkl`, and `label_encoder.pkl`) should be in the same directory as the Streamlit app or provide the correct path.

