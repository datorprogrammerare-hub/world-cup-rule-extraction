# World Cup 2026 Rule Extraction

This project is an interactive Streamlit application that predicts possible FIFA World Cup 2026 contenders using machine learning and rule extraction.

## Project Goal

The goal of this project is to demonstrate how rule extraction can make black-box machine learning models easier to understand.

The application uses a Random Forest model to make predictions and a Decision Tree surrogate model to extract readable IF-THEN rules.

## Features

- Interactive Streamlit dashboard
- FIFA World Cup 2026 team prediction
- Random Forest black-box model
- Decision Tree surrogate model
- Extracted IF-THEN rules
- football-data.org API integration
- Local backup dataset
- Group standings
- Selected country analysis
- Top 10 favorites
- Predicted semifinals and final
- Anthropic AI explanation
- GenAI literacy explanation

## Data Sources

The project uses two data sources:

1. football-data.org API  
2. Local backup dataset

The local backup dataset is used if the API is unavailable.

## Machine Learning Models

### Random Forest

The Random Forest model is used as the main black-box prediction model. It predicts whether a team is a strong candidate for stages such as champion, finalist, or semifinalist.

### Decision Tree Surrogate

A Decision Tree surrogate model is trained to imitate the Random Forest model. This makes it possible to extract readable IF-THEN rules and explain the prediction.

## Features Used by the Model

The model uses football-related features such as:

- FIFA ranking
- Elo rating
- Recent form
- Overall rating
- Attack rating
- Defense rating
- World Cup experience
- Group difficulty
- Points
- Goals for
- Goals against
- Goal difference

## Rule Extraction

The app extracts IF-THEN rules from the Decision Tree surrogate model. These rules help explain why the machine learning model predicts certain teams as stronger candidates.

Example:

```text
IF Elo rating is high AND recent form is strong
THEN the team is more likely to be a champion candidate