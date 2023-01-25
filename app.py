import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(page_title="Disease prediction", page_icon=":guardsman:", layout="wide")
# st.beta_set_theme('material')
# st.beta_set_page_config(page_title="Disease prediction",
#                        page_icon=":guardsman:",
#                        layout="wide",
#                        initial_sidebar_state='auto',
                       # css=
                       # .app .block-container {
                       #      max-width: 1080px;
                       # }
                       # .app .header {
                       #      font-size: 2em;
                       # }
                       # )
st.title('Disease prediction')


age = st.slider("Select age",min_value=0, max_value=120, value=19)
gender = st.radio("Select gender", ['Male','Female','Trans','Gender neutral','Non-binary'])

train_df = st.cache(pd.read_csv)('Training.csv')
test_df = st.cache(pd.read_csv)('Testing.csv')


train_df = train_df.drop(columns=['Unnamed: 133'])

knn = KNeighborsClassifier(n_neighbors = 5)
x_train, y_train = train_df.loc[:,train_df.columns != "prognosis"], train_df.loc[:,"prognosis"]
x_test, y_test = test_df.loc[:,train_df.columns != "prognosis"], test_df.loc[:,"prognosis"]
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

st.write('Having common symptoms can give varied results, please be specific')
X_test_symptoms = np.zeros((1, x_train.shape[1]))
symptoms = []
i = 0
for i in range(3):
    symptom = st.selectbox("Select symptom", list(train_df.columns), key=f'symptom_{i}')
    symptoms.append(symptom)

if st.button('Add more symptoms'):
    symptom = st.selectbox("Select symptom", list(train_df.columns))
    symptoms.append(symptom)
disease_dos_donts = {
        'Fungal infection': '\n + Do:1. Keep hands clean.<br> 2.  Don\'t: Scratch or rub the affected area.',
        'Allergy': '\n + Do: Avoid triggers and take anti-allergic medication. \n + Don\'t: Ignore symptoms and delay treatment.',
        'GERD': '\n + Do: Eat small, frequent meals. Don\'t: Lie down immediately after eating.',
        'Chronic cholestasis': '\n + Do: Avoid high-fat foods. Don\'t: Ignore symptoms and delay treatment.',
        'Drug Reaction': '\n + Do: Discontinue the use of the offending drug. Don\'t: Self-medicate.',
        'Peptic ulcer diseae': '\n + Do: Avoid spicy and oily foods. Don\'t: Delay treatment.',
        'AIDS': '\n + Do: Practice safe sex and take medication as prescribed. Don\'t: Share needles or personal items.',
        'Diabetes': '\n + Do: Monitor blood sugar levels and take medication as prescribed. Don\'t: Ignore symptoms and delay treatment.',
        'Gastroenteritis': '\n + Do: Stay hydrated and eat light, easily digestible foods. Don\'t: Delay treatment or self-medicate.',
        'Bronchial Asthma': '\n + Do: Avoid triggers and take medication as prescribed. Don\'t: Ignore symptoms and delay treatment.',
        'Hypertension': '\n + Do: Eat a healthy diet, exercise regularly and take medication as prescribed. Don\'t: Ignore symptoms and delay treatment.',
        'Migraine': '\n + Do: Identify triggers and avoid them. \n + treatment or self-medicate.',
	   'Hepatitis C': '\n + Do:Stay as physically active as you can by walking, cycling, swimming, gardening, or yoga. \n + Don\'t:Avoid drinking alcoholic beverages.',
	   'Hepatitis E' : '\n + Do:Ensure that your surrounding is hygienic. \n + Don\'t:Limit foods containing saturated fats including fatty cuts of meat and foods fried in oil.',
	  'Alcoholic hepatitis': '\n + Do: Eat a well-balanced, high-calorie diet. People who drink too much alcohol are often malnourished. \n + Don\'t: Avoid salt in your food. Eating salt can make swelling of the abdomen worse.',
 	  'Tuberculosis': '\n + Do:Eat a healthy diet. \n + Don\'t:Stay in a non-ventilated room.',
	  'Common Cold': '\n + Do: Leave salty foods, alcohol, coffee and sugary drinks, which can be dehydrating. Ice chips are another simple way to stay hydrated and calm a scratchy throat. \n+ Don\'t: Avoid Zinc. There’s little evidence to support zinc’s cold-fighting reputation.',
        'Pneumonia' :'\n + Do:Sleep in a partly upright position at night. Place a few pillows under your head or sleep on a reclining chair. \n + Don\'t:Do not use any products that contain nicotine or tobacco, such as cigarettes, e-cigarettes, and chewing tobacco.',
        'Dimorphic hemmorhoids(piles)' : '\n + Do:eat more grains, fruits, and vegetables \n + Don\'t: become physically inactive.',
	  'Heart attack': '\n + Do:Do try to reduce your stress. \n + Don\'t:Avoid use of tobacco. If you smoked or used tobacco, don’t go back to it.',
        'Varicose veins' : '\n + Do:Maintain a consistent diet which is low in sugar, fat and cholesterol. And high in fruit and vegetables. \n+ Don\'t:Over-strain yourself. As tempting as it is to push yourself, if you’re lifting heavy weights or engaging in high impact activity, you can actually damage the vein.',
        'Hypothyroidism': '\n + Do:Take the correct amount of iodine. \n+ Don\'t:Eat excess sugary foods.',
        'Hyperthyroidism': '\n + Do:Eat food which do not have an excess of iodine. \n+ Don\'t:Avoid junk food, alcohol, red meat, carbonated drinks, caffeine and foods having high level of sugar.',
        'Hypoglycemia': '\n + Do:if you experience low blood sugar levels, sit and rest for a while. \n+ Don\'t:Never opt for protein over carbs during a case of Hypoglycemia.',
        'Osteoarthristis': '\n + Do:Choose Low-Impact, Knee Joint-Friendly Exercises. \n + Don\'t:Engage in Repetitive, High-Impact Exercises That Can Harm Your Joints.',
        'Arthritis':'\n + Do:Keep your joints moving. Do daily, gentle stretches that move your joints through their full range of motion.\n + Don\'t:If you\'re addicted to tobacco you might use it as an emotional coping tool. But it\'s counterproductive: Toxins in smoke cause stress on connective tissue, leading to more joint problems.',
        '(vertigo) Paroymsal  Positional Vertigo': '\n + Do:sleep with your head slightly raised on two or more pillows. \n + Don\'t:Never use very high doses of aspirin. Aspirin may cause vertigo when used in high doses.',
        'Acne': '\n + Do:Keep your skin clean. Gently wash your face up to twice daily and after sweating. \n+ Don\'t:Don\'t over-exfoliate.',
        'Urinary tract infection': '\n + Do: DO drink 6 to 8 glasses of water daily. Drinking water and cranberry juice may help the treatment of UTIs. \n+ Don\'t: Avoid drink caffeinated beverages or alcohol.',
        'Psoriasis': '\n + Do:Do moisturize and take a soak. \n + Don\'t: Never ignore flare-ups',
        'Hepatitis D': '\n + Do:You should also avoid alcohol because it can cause more liver damage.\n+ Don\'t: Avoid donating blood.',
        'Hepatitis B' : '\n + Do:DO get plenty of rest and eat a well-balanced diet \n + Don\'t:Smoking is harmful for your health and should be avoided.',
        'hepatitis A': '\n + Do:Drink clean drinking water \n+ Don\'t: Never eat food from street vendors.',
        'Cervical spondylosis' :'\n + Do:Watch your posture while sitting, working, sleeping etc. Correct your posture. \n + Don\'t:Avoid do strenuous activities until your health care provider says you can.',
        'Paralysis (brain hemorrhage)': '\n + Do:Drive carefully, and wear your seat belt. \n + Don\'t:Don’t use drugs. Cocaine, for example, can increase the risk of bleeding in the brain.',
        'Jaundice': '\n + Do:The patient must keep on taking liquid substances like Glucose, Lemon Juice, Dehydrated water in a small quantum frequently. \n + Don\'t:The patient must not talk to someone for a lomg time.',
        'Malaria': '\n + Do:Eat healthy meals and stay hydrated. \n + Don\'t:Avoid  fatty, spicy foods during malaria.',
        'Chicken pox': ' \n + Do:Do monitor the rash closely. \n+ Don\'t: Picking at scratch blisters or scabs.',
        'Dengue':' \n + Do: Stay hydrated and have beetle leaf juice. \n+ Don\'t: Avoid going into public ares and open spaces.',
        'Typhoid':  '\n + Do:Always drink filtered and boiled mineral water with the proper hygiene value \n + Don\'t:Foods which are high in fibre are the ones that should be limited in the diet of a person suffering from typhoid.',
        'Impetigo': '\n + Do:Gently wash the affected areas with mild soap and running water and then cover lightly with gauze. \n + Don\'t: Avoid direct skin-to-skin contact with others.'}
if st.button('Predict'):
    for symptom in symptoms:
        if symptom in list(train_df.columns):
            X_test_symptoms[0, list(train_df.columns).index(symptom)] = 1
    y_pred = knn.predict(X_test_symptoms)
    st.success("Predicted Prognosis: " + y_pred[0])
    st.write("Do's and Don'ts: ", disease_dos_donts.get(y_pred[0], "Not available"))
    result = st.checkbox("Have you been diagnosed with {}?".format(y_pred[0]))
    if result:
        st.write("It's highly recommended that you see a doctor.")
    else:
        st.write("Thank you for letting us know.")
st.empty()
st.empty()
with st.container():
    st.markdown("---")
    st.write("Tinker assumes no responsibility or liability for any errors or omissions in the content of this site. The information contained in this site is provided on an  basis with no guarantees of completeness, accuracy, usefulness or timeliness...")
    st.markdown("Copyright © Tinker")