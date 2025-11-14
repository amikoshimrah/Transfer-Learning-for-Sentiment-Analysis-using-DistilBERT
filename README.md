📌 Problem Statement

Sentiment analysis usually requires large datasets and long training cycles, which makes training models from scratch inefficient. Transfer Learning provides a solution by using pre-trained language models that already understand the structure of human language.

This project aims to fine-tune DistilBERT to classify text sentiment. It builds a full pipeline that takes raw text, processes it, trains a transformer-based model and evaluates it on unseen data.

📊 Project Summary

This notebook walks through a full workflow for building a sentiment classifier using DistilBERT. It starts with data preparation, then tokenizes the text with HuggingFace transformers and converts everything into TensorFlow-ready datasets. The model is trained using transfer learning, and performance is evaluated using accuracy, classification reports and a confusion matrix. The project ends by saving the fine-tuned model and tokenizer so they can be reused for inference or deployment.

🎯 Objectives

Prepare text data for training, validation and testing

Tokenize text using a DistilBERT tokenizer

Convert datasets into TensorFlow pipelines

Fine-tune a DistilBERT sequence classification model

Evaluate results using standard NLP classification metrics

Visualize performance using a confusion matrix

Save the trained model and tokenizer for future use

🧩 Workflow Breakdown
1. Data Preparation

Reads train, validation and test DataFrames

Ensures each entry has text and label fields

Converts them into HuggingFace Datasets

2. Tokenization

Uses AutoTokenizer

Applies truncation and padding

Converts text → input IDs + attention masks

3. Dataset Conversion

Uses DataCollatorWithPadding for efficient batching

Builds TensorFlow datasets for fast training

4. Model Building & Fine-Tuning

Loads distilbert-base-uncased with a 2-label classification head

Compiles the model with Adam optimizer and cross-entropy loss

Supports early stopping and checkpoint saving

Optionally uses class weighting for imbalanced datasets

5. Evaluation

Predicts labels on the test dataset

Computes:

Accuracy

Precision, Recall, F1 Score

A full classification report

Plots a confusion matrix to visualize prediction quality

6. Saving the Model

The final DistilBERT model and tokenizer are saved locally, making it easy to load them later for inference:

./distilbert_sentiment_model/

🧠 Key Learnings

How transformer tokenizers process raw text

How to fine-tune a pre-trained model for classification

How to build TensorFlow pipelines from HF datasets

The importance of model evaluation using real metrics

How to save and reuse transformers for downstream tasks
