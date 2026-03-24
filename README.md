# Content Suggestion System (Machine Learning)

A simple **Content Recommendation System** built using **Random Forest** and **Support Vector Machine (SVM)**.
This project predicts whether a user will like a piece of content based on structured features and text data.

## Features

*  Random Forest for feature-based recommendations
*  SVM for text-based recommendations
*  TF-IDF vectorization for content understanding
*  Easy to extend and integrate into real applications

---

## Project Structure

content-suggestion/
1. main.py
2. requirements.txt
3. README.md

## Requirements

scikit-learn
numpy
pandas


#### How It Works

### 1. Random Forest Model

* Uses numerical features:

  * Interest match (0 or 1)
  * Popularity score (0–1)
  * Content length
* Predicts whether a user will like content

### 2. SVM Model

* Uses text data (content titles/descriptions)
* Converts text into vectors using **TF-IDF**
* Classifies content as "liked" or "not liked"



##  Usage

Run the project:

bash
python main.py


##  Example Output

=== Content Suggestion System ===

**Random Forest Prediction
Features: [1, 0.8, 6]
Will user like? Yes
Confidence: 0.87

** SVM Text Prediction
Text: python machine learning tutorial
Will user like? Yes
Confidence: 0.91

##  Workflow

1. Prepare dataset (user behavior or content features)
2. Train machine learning models
3. Predict user preferences
4. Recommend relevant content

## Future Improvements

*  Add real user data (clicks, watch time)
*  Build REST API using Flask or FastAPI
*  Integrate with frontend (React / mobile app)
*  Use deep learning (neural networks)
*  Deploy on cloud (AWS / GCP)

##  Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

##  License

This project is licensed under the **MIT License**.
