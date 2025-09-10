
import gradio as gr
import joblib
import pandas as pd

#  الموديل اللي لسه حفظتيه
model = joblib.load("grid_search_rf.pkl")
scaler = joblib.load("scaler.pkl")


# أعمدة الداتا  بالأسامي الصحيحة وبنفس الترتيب
feature_names = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time"
]

def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes,
                          ejection_fraction, high_blood_pressure,
                          platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time):

    # حول اختيارات Yes/No و Male/Female لأرقام
    anaemia = 1 if anaemia == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
    sex = 1 if sex == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0

    # حول الأرقام لأنواع رقمية
    age = float(age)
    creatinine_phosphokinase = float(creatinine_phosphokinase)
    ejection_fraction = float(ejection_fraction)
    platelets = float(platelets)
    serum_creatinine = float(serum_creatinine)
    serum_sodium = float(serum_sodium)
    time = float(time)

    # جهّز DataFrame بنفس الأعمدة والترتيب
    row = [[age, anaemia, creatinine_phosphokinase, diabetes,
            ejection_fraction, high_blood_pressure,
            platelets, serum_creatinine, serum_sodium,
            sex, smoking, time]]

    X = pd.DataFrame(row, columns=feature_names)


    pred = model.predict(X)[0]

    # لو الموديل بيدعم الاحتمالات، نعرض الثقة
    proba_txt = ""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0][1]
        proba_txt = f" (prob={p:.2f})"

    return ("🚨 High risk of heart disease" if pred == 1 else "✅ Your results are normal") + proba_txt





with gr.Blocks() as demo:
    gr.Markdown("## 🫀 Heart Failure Prediction App")

    with gr.Row():
        age = gr.Number(label="Age")
        anaemia = gr.Radio(["Yes","No"], label="Anaemia")
        creatinine_phosphokinase = gr.Number(label="Creatinine Phosphokinase")
        diabetes = gr.Radio(["Yes","No"], label="Diabetes")
        ejection_fraction = gr.Number(label="Ejection Fraction")
        high_blood_pressure = gr.Radio(["Yes","No"], label="High Blood Pressure")

    with gr.Row():
        platelets = gr.Number(label="Platelets")
        serum_creatinine = gr.Number(label="Serum Creatinine")
        serum_sodium = gr.Number(label="Serum Sodium")
        sex = gr.Radio(["Male","Female"], label="Sex")
        smoking = gr.Radio(["Yes","No"], label="Smoking")
        time = gr.Number(label="Time (days)")

    btn = gr.Button("Predict")
    out = gr.Textbox(label="Result")

    btn.click(
        predict_heart_failure,
        inputs=[age, anaemia, creatinine_phosphokinase, diabetes,
                ejection_fraction, high_blood_pressure,
                platelets, serum_creatinine, serum_sodium,
                sex, smoking, time],
        outputs=out
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)