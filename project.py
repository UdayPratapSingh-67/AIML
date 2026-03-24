
Content Suggestion System

Uses:
1. Random Forest (feature-based recommendation)
2. SVM (text-based recommendation)

Run:
    python main.py


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


#  RANDOM FOREST (Feature-Based Model)

class RandomForestRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        # Features: [interest_match, popularity_score, content_length]
        self.X = [
            [1, 0.9, 5],
            [0, 0.2, 10],
            [1, 0.7, 6],
            [0, 0.3, 8],
            [1, 0.85, 4],
            [0, 0.1, 12]
        ]

        # Labels: 1 = liked, 0 = not liked
        self.y = [1, 0, 1, 0, 1, 0]

        self.model.fit(self.X, self.y)

    def predict(self, features):
        pred = self.model.predict([features])[0]
        prob = self.model.predict_proba([features])[0][1]
        return pred, prob



# SVM (Text-Based Model)


class SVMRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)

    def train(self):
        self.texts = [
            "learn python programming",
            "advanced machine learning",
            "web development html css",
            "deep learning ai",
            "javascript frontend basics",
            "data science with python"
        ]

        self.labels = [1, 1, 0, 1, 0, 1]

        X = self.vectorizer.fit_transform(self.texts)
        self.model.fit(X, self.labels)

    def predict(self, text):
        X_new = self.vectorizer.transform([text])
        pred = self.model.predict(X_new)[0]
        prob = self.model.predict_proba(X_new)[0][1]
        return pred, prob



#  MAIN FUNCTION


def main():
    print("\n=== Content Suggestion System ===\n")

    # Initialize models
    rf = RandomForestRecommender()
    svm = SVMRecommender()

    # Train models
    rf.train()
    svm.train()

    # -------- Random Forest Example --------
    print("🔹 Random Forest Recommendation")
    features = [1, 0.8, 6]  # Example input
    pred, prob = rf.predict(features)

    print(f"Input Features: {features}")
    print("Prediction:", "👍 Like" if pred == 1 else "👎 Not Like")
    print(f"Confidence: {prob:.2f}\n")

    # -------- SVM Example --------
    print("🔹 SVM Text Recommendation")
    text = "python machine learning tutorial"
    pred, prob = svm.predict(text)

    print(f"Input Text: {text}")
    print("Prediction:", "👍 Like" if pred == 1 else "👎 Not Like")
    print(f"Confidence: {prob:.2f}\n")


#  ENTRY POINT


if __name__ == "__main__":
    main()
