🕵️ Dark Patterns Detection in E-Commerce

An automated, ML-powered system to detect manipulative UI/UX design patterns on e-commerce platforms using Natural Language Processing and Machine Learning.

👥 Authors
Devansh Pundir
Somay Agarwal
Devansh Srivastava

📍 JK Lakshmipat University | Course: CS1138 – Machine Learning
📖 Overview
E-commerce platforms increasingly exploit dark patterns — deceptive UI/UX design techniques that manipulate users into unintended actions such as impulsive purchases, unwanted subscriptions, or unnecessary data sharing. This project builds an automated, context-aware detection system using classical Machine Learning and TF-IDF text features to identify such patterns at scale.
The system achieves a 0.98 AUC score with Random Forest, making it lightweight and accurate enough for real-time deployment in browser extensions.

❗ Problem Statement
E-commerce platforms use dark patterns like:

Fake urgency – "Only 2 left in stock!"
Hidden costs – Taxes/fees revealed only at checkout
Misleading prompts – Pre-checked boxes for unwanted subscriptions

Manual detection of these practices is:

Time-consuming and impractical at scale
Inconsistent across reviewers

Existing keyword-based methods lack contextual understanding, creating the need for an intelligent, automated solution.

💡 Motivation

Dark patterns push users into impulsive purchases or unwanted sign-ups, reducing trust in digital platforms.
There is currently no scalable, automated solution for real-time dark pattern detection during browsing.
This project addresses that gap by combining NLP feature extraction with machine learning classification.


📊 Dataset
PropertyDetailTotal Samples2,356Feature TypeTextTarget Label1 (Dark Pattern) / 0 (Not Dark Pattern)Data TypeText DataClass DistributionBalanced
The balanced class distribution ensures the model learns genuine distinguishing features of both classes without bias.

⚙️ Methodology
The project follows a 6-step pipeline:
Raw Data → Preprocessing → TF-IDF Vectorization → Model Training → Model Evaluation → Prediction
Step-by-step:

Raw Data – Collect labeled e-commerce text snippets (product pages, buttons, banners, checkout flows)
Preprocessing – Lowercasing, punctuation removal, stop word removal, tokenization, stemming/lemmatization
TF-IDF Vectorization – Convert cleaned text into numerical feature vectors using Term Frequency–Inverse Document Frequency
Model Training – Train three ML classifiers on the feature vectors
Model Evaluation – Compare models using Accuracy, Precision, Recall, F1 Score, and AUC
Prediction – Deploy the best model for real-time dark pattern classification


🧠 Models Used
Three classifiers were trained and compared:
#ModelKey Strength1Logistic RegressionSimple, interpretable, fast2Support Vector Machine (SVM)Effective for high-dimensional text data3Random Forest ✅Robust, handles non-linear relationships, ensemble-based

📈 Results
ModelAccuracyPrecisionRecallF1 ScoreAUC ScoreLogistic Regression92.58%0.97180.87710.92200.9756Support Vector Machine93.01%0.96350.89410.92750.9704Random Forest93.64%0.95980.91100.93480.9793
🏆 Best Model: Random Forest
Confusion Matrix (Random Forest):
                  Predicted: Not Dark    Predicted: Dark
Actual: Not Dark       227                    9
Actual: Dark            21                  215

True Positives: 215 | True Negatives: 227
False Positives: 9 | False Negatives: 21
ROC-AUC: ~0.98


🔍 Key Insights
1. Optimal Balance
The high F1-score (0.9348) proves that Random Forest effectively catches deceptive patterns (high recall) while minimizing false alarms for honest websites (high precision).
2. Linguistic Triggers
Feature importance analysis identified the following words as the strongest indicators of dark patterns:

"only" — artificial scarcity
"left" — fake limited stock
"hurry" — false urgency
"expires" — fake countdown pressure

3. Classical ML beats Deep Learning
Random Forest outperformed DistilBERT (a transformer model), demonstrating that TF-IDF statistical keyword analysis is more effective than deep semantic context for this dataset. This is because dark patterns rely on specific trigger words, not nuanced semantics — making TF-IDF the ideal feature extractor.

🛒 Dark Pattern Examples
TypeExampleArtificial Scarcity"Flash Sale! Limited Time Only — Shop Now!"Hidden CostsSubtotal: $50.00 → Tax & Fees: $10.51 → Total: $62.00 (revealed at checkout)Fake Countdown Timers00:04:59 countdown shown on software subscription page

🚀 Future Work

 Chrome Extension – Extend the system into a Google Chrome browser extension for real-time detection of dark patterns on e-commerce websites during browsing.
 Pattern Categorization – Identify and classify specific dark pattern types (fake urgency, hidden costs, roach motel, etc.) rather than binary detection.
 User Alerts – Alert users with clear explanations when dark patterns are detected on a page.
 Multilingual Support – Extend the model to support regional languages (Hindi, Tamil, etc.) for Indian e-commerce platforms.
 Active Learning Pipeline – Allow users to flag missed dark patterns to continuously improve the model.
