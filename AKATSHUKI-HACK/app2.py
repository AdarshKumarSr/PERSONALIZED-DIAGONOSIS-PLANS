import pandas as pd

def dict():
    mentaldata = {
    'Symptoms': [
        'Persistent sadness, loss of interest',
        'Excessive worry, restlessness, difficulty concentrating',
        'Hallucinations, delusions, disorganized thinking',
        'Mood swings, irritability, impulsive behavior',
        'Intense fear, panic attacks, avoidance behaviors',
        'Intrusive thoughts, repetitive behaviors',
        'Social withdrawal, lack of emotional expression',
        'Extreme mood changes, unstable self-image',
        'Excessive eating or dieting behaviors',
        'Difficulty sleeping, nightmares, hyperarousal'
    ],
    'Disease': [
        'Depression',
        'Generalized Anxiety Disorder',
        'Schizophrenia',
        'Bipolar Disorder',
        'Panic Disorder',
        'Obsessive-Compulsive Disorder',
        'Schizoid Personality Disorder',
        'Borderline Personality Disorder',
        'Eating Disorders',
        'Post-Traumatic Stress Disorder'
    ],
    'Suggestions': [
        'Counseling, medication, exercise',
        'Cognitive behavioral therapy, relaxation techniques',
        'Antipsychotic medication, therapy',
        'Mood stabilizers, therapy, sleep management',
        'Breathing exercises, exposure therapy',
        'Exposure and response prevention, medication',
        'Social skills training, therapy',
        'Dialectical behavior therapy, medication',
        'Nutritional counseling, therapy, support groups',
        'Trauma-focused therapy, relaxation techniques'
    ]
}

dict()

df = pd.DataFrame(mentaldata)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df['Disease_Label'] = le.fit_transform(df['Disease'])

X_train, X_test, y_train, y_test = train_test_split(df['Symptoms'], df['Disease_Label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_vec, y_train)

y_pred = svm.predict(X_test_vec)

labels = list(le.classes_)
print(classification_report(y_test, y_pred, labels=le.transform(labels), target_names=labels))

def predict_disease_and_suggestions(symptoms):
    symptoms_vec = vectorizer.transform([symptoms])
    disease_label = svm.predict(symptoms_vec)[0]
    disease = le.inverse_transform([disease_label])[0]
    suggestions = df[df['Disease'] == disease]['Suggestions'].iloc[0]
    return disease, suggestions

# Example usage
input_symptoms = "persistent sadness, lack of interest"
predicted_disease, suggested_treatments = predict_disease_and_suggestions(input_symptoms)
print(f"Predicted Disease: {predicted_disease}")
print(f"Suggestions: {suggested_treatments}")