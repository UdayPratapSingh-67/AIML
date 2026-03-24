def main():
print("=== Content Suggestion System ===\n")

# Train models
rf_model = train_random_forest()
svm_model, vectorizer = train_svm()

# Example input for Random Forest
print("🔹 Random Forest Prediction")
rf_features = [1, 0.8, 6]  # Example: matches interest, popular, medium length
rf_pred, rf_prob = predict_with_rf(rf_model, rf_features)

print(f"Features: {rf_features}")
print(f"Will user like? {'Yes' if rf_pred == 1 else 'No'}")
print(f"Confidence: {rf_prob:.2f}\n")

# Example input for SVM
print("🔹 SVM Text Prediction")
text_input = "python machine learning tutorial"
svm_pred, svm_prob = predict_with_svm(svm_model, vectorizer, text_input)

print(f"Text: {text_input}")
print(f"Will user like? {'Yes' if svm_pred == 1 else 'No'}")
print(f"Confidence: {svm_prob:.2f}\n")

if name == "main":
main()
