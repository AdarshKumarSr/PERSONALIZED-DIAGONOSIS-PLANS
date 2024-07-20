from flask import Flask,render_template,url_for, redirect, session, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from langchain import PromptTemplate, LLMChain
import pyttsx3
import os

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')


app = Flask(__name__)

# Symptoms Checker:
load_dotenv(find_dotenv())
HUGGINGFACEHUB_APT_TOKEN = os.getenv("HUGGINGFACEHUB_APT_TOKEN")

def suggesions(prog):
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=20,temperature=0.7,token="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")
    return llm.invoke(f"What is {prog}? and what should i do if i have one?, do i need medical check up or not?")


def processedtext(prog):
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=20,temperature=0.7,token="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")
    return llm.invoke(f"summerize the following text in diagnosis and prescription and also symtoms: "+ str(prog))

symptoms_dict = {"itching":"Itching","skin_rash":"Skin rash","nodal_skin_eruptions":"Nodal skin eruptions","continuous_sneezing":"Continuous sneezing","shivering":"Shivering","chills":"Chills","joint_pain":"Joint pain","stomach_pain":"Stomach pain","acidity":"Acidity","ulcers_on_tongue":"Ulcers on tongue","muscle_wasting":"Muscle wasting","vomiting":"Vomiting","burning_micturition":"Burning micturition","spotting_ urination":"Spotting urination","fatigue":"Fatigue","weight_gain":"Weight gain","anxiety":"Anxiety","cold_hands_and_feets":"Cold hands and feets","mood_swings":"Mood swings","weight_loss":"Weight loss","restlessness":"Restlessness","lethargy":"Lethargy","patches_in_throat":"Patches in throat","irregular_sugar_level":"Irregular sugar level","cough":"Cough","high_fever":"High fever","sunken_eyes":"Sunken eyes","breathlessness":"Breathlessness","sweating":"Sweating","dehydration":"Dehydration","indigestion":"Indigestion","headache":"Headache","yellowish_skin":"Yellowish skin","dark_urine":"Dark urine","nausea":"Nausea","loss_of_appetite":"Loss of appetite","pain_behind_the_eyes":"Pain behind the eyes","back_pain":"Back pain","constipation":"Constipation","abdominal_pain":"Abdominal pain","diarrhoea":"Diarrhoea","mild_fever":"Mild fever","yellow_urine":"Yellow urine","yellowing_of_eyes":"Yellowing of eyes","acute_liver_failure":"Acute liver failure","fluid_overload":"Fluid overload","swelling_of_stomach":"Swelling of stomach","swelled_lymph_nodes":"Swelled lymph nodes","malaise":"Malaise","blurred_and_distorted_vision":"Blurred and distorted vision","phlegm":"Phlegm","throat_irritation":"Throat irritation","redness_of_eyes":"Redness of eyes","sinus_pressure":"Sinus pressure","runny_nose":"Runny nose","congestion":"Congestion","chest_pain":"Chest pain","weakness_in_limbs":"Weakness in limbs","fast_heart_rate":"Fast heart rate","pain_during_bowel_movements":"Pain during bowel movements","pain_in_anal_region":"Pain in anal region","bloody_stool":"Bloody stool","irritation_in_anus":"Irritation in anus","neck_pain":"Neck pain","dizziness":"Dizziness","cramps":"Cramps","bruising":"Bruising","obesity":"Obesity","swollen_legs":"Swollen legs","swollen_blood_vessels":"Swollen blood vessels","puffy_face_and_eyes":"Puffy face and eyes","enlarged_thyroid":"Enlarged thyroid","brittle_nails":"Brittle nails","swollen_extremeties":"Swollen extremeties","excessive_hunger":"Excessive hunger","extra_marital_contacts":"Extra marital contacts","drying_and_tingling_lips":"Drying and tingling lips","slurred_speech":"Slurred speech","knee_pain":"Knee pain","hip_joint_pain":"Hip joint pain","muscle_weakness":"Muscle weakness","stiff_neck":"Stiff neck","swelling_joints":"Swelling joints","movement_stiffness":"Movement stiffness","spinning_movements":"Spinning movements","loss_of_balance":"Loss of balance","unsteadiness":"Unsteadiness","weakness_of_one_body_side":"Weakness of one body side","loss_of_smell":"Loss of smell","bladder_discomfort":"Bladder discomfort","foul_smell_of urine":"Foul smell of urine","continuous_feel_of_urine":"Continuous feel of urine","passage_of_gases":"Passage of gases","internal_itching":"Internal itching","toxic_look_(typhos)":"Toxic look(typhos)","depression":"Depression","irritability":"Irritability","muscle_pain":"Muscle pain","altered_sensorium":"Altered sensorium","red_spots_over_body":"Red spots over body","belly_pain":"Belly pain","abnormal_menstruation":"Abnormal menstruation","dischromic _patches":"Dischromic patches","watering_from_eyes":"Watering from eyes","increased_appetite":"Increased appetite","polyuria":"Polyuria","family_history":"Family history","mucoid_sputum":"Mucoid sputum","rusty_sputum":"Rusty sputum","lack_of_concentration":"Lack of concentration","visual_disturbances":"Visual disturbances","receiving_blood_transfusion":"Receiving blood transfusion","receiving_unsterile_injections":"Receiving unsterile injections","coma":"Coma","stomach_bleeding":"Stomach bleeding","distention_of_abdomen":"Distention of abdomen","history_of_alcohol_consumption":"History of alcohol consumption","fluid_overload1":"Fluid overload1","blood_in_sputum":"Blood in sputum","prominent_veins_on_calf":"Prominent veins on calf","palpitations":"Palpitations","painful_walking":"Painful walking","pus_filled_pimples":"Pus filled pimples","blackheads":"Blackheads","scurring":"Scurring","skin_peeling":"Skin peeling","silver_like_dusting":"Silver like dusting","small_dents_in_nails":"Small dents in nails","inflammatory_nails":"Inflammatory nails","blister":"Blister","red_sore_around_nose":"Red sore around nose","yellow_crust_ooze":"Yellow crust ooze"}


list_ = [item for item in symptoms_dict.values() if item]
common_phrases = [item for sublist in list_ for item in sublist.split(' and ') if item]


# Function to extract symptoms from text
def extract_symptoms(text):
    # Create a pattern to match these phrases in the text
    pattern = '|'.join(map(re.escape, common_phrases))

    # Find and extract these phrases from the text
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    # Remove matched phrases from the original text to avoid duplication
    for match in matches:
        text = re.sub(re.escape(match), '', text, flags=re.IGNORECASE)

    # Tokenize the remaining text
    tokens = word_tokenize(text)

    # Remove punctuation and stop words from the remaining tokens
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatize the remaining tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Combine the matches and lemmatized tokens, removing duplicates
    symptoms = list(set([match.capitalize() for match in matches] + [token.capitalize() for token in lemmatized_tokens]))

    return symptoms

data = pd.read_csv("Training.csv")

X = data.drop(['prognosis'], axis=1).values
Y = data['prognosis'].values

model = LogisticRegression()
model.fit(X, Y)

def prognosis(input:str):
    input_list = extract_symptoms(input)

    # Create a list to store the binary representation
    binary_representation = []

    # Iterate through the symptoms dictionary values
    for symptom in symptoms_dict.values():
        if symptom in input_list:
            binary_representation.append(1)
        else:
            binary_representation.append(0)

    # Convert the list to a numpy array with 1 row and len(symptoms_dict) columns
    output_array = np.array(binary_representation).reshape(1, len(symptoms_dict))
    prognosis_prediction = model.predict(output_array)
    return prognosis_prediction

def speechRecogition():
    
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as mic:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                text = recognizer.recognize_google(audio)
                text = text.lower()

                if text:
                    return text
        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError as e:
            print(f"Could not request results; {e}")

        except KeyboardInterrupt:
            break

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


df = pd.DataFrame(mentaldata)
le = LabelEncoder()
df['Disease_Label'] = le.fit_transform(df['Disease'])

X_train, X_test, y_train, y_test = train_test_split(df['Symptoms'], df['Disease_Label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

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


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sympChecker", methods = ['GET', 'POST'])
def sympChecker():
    if request.method == 'POST':
        text = str(request.form["symptons"])
        prog = prognosis(text)
        info = suggesions(prog[0])
        return render_template("sympChecker.html", prog = prog[0], info=info)
    return render_template("sympChecker.html")

@app.route("/mentalHealth", methods = ['GET', 'POST'])
def mentalHealth():
    if request.method == 'POST':
        text = str(request.form["symptons"])
        prog,info =  predict_disease_and_suggestions(text)
        return render_template("mentalHealth.html", prog = prog, info=info)
    return render_template("mentalHealth.html")

@app.route("/speechReck", methods = ['GET', 'POST'])
def speechReck():
    text = speechRecogition()
    return render_template("sympChecker.html", text = text)

@app.route("/docassist", methods = ['GET', 'POST'])
def docassist():
    processed_text = ''
    if request.method == 'POST':
        text = str(request.form["symptons"])
        processed_text = processedtext(text)
    return render_template("docassist.html", info = processed_text)

@app.route("/speechrec", methods = ['GET', 'POST'])
def speechrec():
    text = speechRecogition()
    return render_template("docassist.html", text = text)



if __name__ == "__main__":
    app.run(debug=True)