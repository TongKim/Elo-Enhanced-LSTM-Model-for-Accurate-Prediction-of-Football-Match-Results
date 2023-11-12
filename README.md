# Elo-Enhanced LSTM Model for Football Match Prediction

## Introduction
This project features a football match prediction framework that integrates LSTM (Long Short-Term Memory) models with the Elo rating system. It aims to predict the number of goals scored in a match using historical match data, with the objective of outperforming bookmakers in the betting market. The model has been trained and tested on English Premier League data, achieving a mean squared error of 1.2293 and an R-squared of 0.14. In simulated betting, the model yields a 5.33% total return, showcasing its potential to outperform bookmakers. However, the model also displays limitations, particularly in predicting outcomes for newly promoted teams and with limited input data.

## Installation Guide
To run this code, you will need to install the following dependencies:
- Python 3
- Pandas
- Numpy
- PyTorch
- Scikit-learn
- Apex (optional, for accelerated training)

These dependencies can be installed using Python's package manager, pip:
```bash
pip install pandas numpy torch scikit-learn
