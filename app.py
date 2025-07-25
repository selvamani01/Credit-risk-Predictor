from flask import Flask, render_template, request, send_from_directory
import os, pickle, pandas as pd
from fpdf import FPDF
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

# Load model, scaler, encoders
model    = pickle.load(open('model.pkl',   'rb'))
scaler   = pickle.load(open('scaler.pkl',  'rb'))
encoders = pickle.load(open('encoders.pkl','rb'))

categorical_cols = [
    'person_home_ownership', 'loan_intent',
    'loan_grade', 'cb_person_default_on_file'
]

PDF_PATH = os.path.join('static', 'report.pdf')

# Utility: Generate PDF
def generate_pdf_report(form_data: dict, prediction: str, path: str = PDF_PATH):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Credit Risk Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for k, v in form_data.items():
        pdf.cell(200, 8, txt=f"{k.replace('_',' ').title()}: {v}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0,150,0) if prediction=="Low Risk" else pdf.set_text_color(220,50,50)
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.output(path)

# Utility: Text to Speech
def speak(txt: str):
    engine = pyttsx3.init()
    engine.say(txt)
    engine.runAndWait()

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_data = {}
        for field in request.form:
            if field == 'submit_prediction':
                continue
            val = request.form[field]
            val = encoders[field].transform([val])[0] if field in categorical_cols else float(val)
            user_data[field] = val

        df_in = pd.DataFrame([user_data])
        scaled = scaler.transform(df_in)
        pred = model.predict(scaled)[0]
        prediction = "High Risk" if pred == 1 else "Low Risk"

        generate_pdf_report(request.form.to_dict(), prediction)

    return render_template('index.html', prediction=prediction)

# Voice input demo
@app.route('/voice')
def voice_demo():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        speak("Please say your monthly income in rupees")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening... Start speaking.")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)

        text = recognizer.recognize_google(audio)
        return f"You said: {text}"
    except Exception as e:
        return f"Voice input error: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
