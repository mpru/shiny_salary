from pathlib import Path
import joblib
from shiny import App, reactive, render, ui
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar los modelos entrenados
modelo_reglin = joblib.load("model_outputs/mod_reg_lin.pkl")
modelo_transf = AutoModelForSequenceClassification.from_pretrained('model_outputs/mod_distilbert')
tokenizer = AutoTokenizer.from_pretrained('model_outputs/tokenizador')

# Función para predecir con el modelo transformer, desde un texto
def predict_transformer(text):
    # Tokenizar el texto
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Asegurarse de usar la GPU si está disponible
    inputs = {key: val.to(modelo_transf.device) for key, val in inputs.items()}
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = modelo_transf(**inputs)
        prediction_log = outputs.logits.squeeze().item()  # Predicción en escala log
        prediction_original = 10 ** prediction_log  # Convertir la predicción a la escala original
        
    return prediction_original

# Función para hacer predicciones con transformer para un DataFrame de pandas
def predict_transformer_from_pd(df):
    
    # Crear una lista para almacenar las predicciones
    predictions = []
    
    # Iterar sobre las filas del DataFrame y predecir el salario
    for _, row in df.iterrows():
        salary_pred = predict_transformer(row['text'])
        predictions.append(salary_pred)

    return np.array(predictions)

# Componente de tarjeta
def ui_card(title, *args):
    return ui.div(
        {"class": "card mb-4"},
        ui.div(title, class_="card-header"),
        ui.div({"class": "card-body"}, *args),
    )


# Definición de la interfaz
app_ui = ui.page_fluid(
    # ui.h2("Making salary predictions for new cases"),
    ui_card(
        "Specifications",
        ui.input_select(
            "modelo",
            "Select the model you want to use",
            choices=["Linear regression model", "Transformers (LLM)"],
        ),
        ui.input_radio_buttons(
            "tipo_prediccion",
            "Data source",
            choices=["Individual case", "Cases from CSV file"],
        ),
    ),
    ui_card(
        "Input",
        ui.output_ui("dynamic_input"),
    ),
    ui.input_action_button(
        "calculate_button", "Calculate predictions",
        style="background-color: #4a4a4a; color: white; margin-bottom: 20px;"
    ),
    ui.output_ui("show_predictions"), # card de predicciones, aparece si se hace clic en el boton
    ui.div(
        "Created by Marcos Prunello with Shiny for Python", 
        align = "center", 
        style = "font-size: small; color: gray;"
    ),
)


# Lógica del servidor
def server(input, output, session):

    # aca guardo el dataframe con predicciones que genera el server, para poder usarlo en otros lugares, es un reactive value
    val = reactive.value(pd.DataFrame())

    # Actualizar la sección de entrada según la fuente seleccionada
    @output
    @render.ui
    def dynamic_input():
        if input.modelo() == "Linear regression model":
            if input.tipo_prediccion() == "Individual case":
                return ui.div(
                    ui.input_numeric("input_age", "Age", value=35),
                    ui.input_numeric("input_exp", "Years of experience", value=10),
                    ui.input_select("input_gender", "Gender", choices=["Female", "Male"]),
                    ui.input_select("input_educ", "Education", choices=["Bachelor's", "Master's", "PhD"]),
                    ui.input_select("input_title_cat", "Choose the word most related to your job title", choices=["Junior", "Senior", "Leadership", "Other"])
                )
            else:
                return ui.div(
                    ui.input_file("file", "Load your CSV file"),
                    ui.download_button("download_example_reg", "Download sample CSV file for the linear regression model")
                )

        elif input.modelo() == "Transformers (LLM)":
            if input.tipo_prediccion() == "Individual case":
                return ui.div(
                    ui.input_text("input_text", "Enter the text")
                )
            else:
                return ui.div(
                    ui.input_file("file", "Load your CSV file (must contain a 'text' column)"),
                    ui.download_button("download_example_llm", "Download sample CSV file for the Transformers model")
                )

    # Descargar un archivo CSV de ejemplo, modelo regresion
    @render.download
    def download_example_reg():
        csv_content = "age,gender,educ,exp,title_cat\n30,Male,Bachelor's,5,Junior\n30,Female,Bachelor's,5,Junior"
        path = Path("sample.csv")
        path.write_text(csv_content)
        return str(path)

    # Descargar un archivo CSV de ejemplo, modelo llm
    @render.download
    def download_example_llm():
        csv_content = "text\nI am a 46-year-old Senior Project Manager with a Master's degree and 19 years of experience in project management.\nI am a 33-year-old male working as a Junior Business Analyst with a Bachelor's degree and four years of experience."
        path = Path("sample.csv")
        path.write_text(csv_content)
        return str(path)

    # Lógica para manejar el botón "Calculate predictions"
    @output
    @render.table
    @reactive.event(input.calculate_button)
    def predictions():
        # Dependiendo de la fuente de datos, calculamos las predicciones
        if input.modelo() == "Linear regression model":
            if input.tipo_prediccion() == "Individual case":
                # Verificar si todos los campos están completos
                if all([input.input_age(), input.input_exp(), input.input_gender(), input.input_educ(), input.input_title_cat()]):
                    data = pd.DataFrame(
                        [[input.input_age(), input.input_gender(), input.input_educ(), input.input_exp(), input.input_title_cat()]],
                        columns=["age", "gender", "educ", "exp", "title_cat"]
                    )
                    # return data
                else:
                    return pd.DataFrame({"Error": ["Please fill all the fields"]})

            elif input.tipo_prediccion() == "Cases from CSV file":
                if input.file() is None:
                    return pd.DataFrame({"Error": ["No file uploaded"]})
                
                # Leer archivo CSV
                file_info = input.file()
                file_path = file_info[0]["datapath"]
                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    return pd.DataFrame({"Error": [f"Error reading CSV file: {e}"]})

                # Validar las columnas requeridas
                required_columns = ["age", "gender", "educ", "exp", "title_cat"]
                if not all(col in data.columns for col in required_columns):
                    return pd.DataFrame({"Error": ["Missing required columns in CSV file"]})

                # return data

        elif input.modelo() == "Transformers (LLM)":
            if input.tipo_prediccion() == "Individual case":
                # Solo se necesita el campo de texto
                if not input.input_text():
                    return pd.DataFrame({"Error": ["Please enter the text"]})
                data = pd.DataFrame({"text": [input.input_text()]})

            elif input.tipo_prediccion() == "Cases from CSV file":
                if input.file() is None:
                    return pd.DataFrame({"Error": ["No file uploaded"]})
                
                # Leer archivo CSV
                file_info = input.file()
                file_path = file_info[0]["datapath"]
                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    return pd.DataFrame({"Error": [f"Error reading CSV file: {e}"]})

                # Validar que el archivo tenga solo la columna "text"
                if "text" not in data.columns:
                    return pd.DataFrame({"Error": ["CSV must contain one column named 'text'"]})

                # return data

        # Seleccionar el modelo y hacer las predicciones
        if input.modelo() == "Linear regression model":
            predictions = modelo_reglin.predict(data)
            data.insert(loc = 0, column = "predicted_salary", value = np.round(predictions))
        elif input.modelo() == "Transformers (LLM)":
            predictions = predict_transformer_from_pd(data)
            data.insert(loc = 0, column = "predicted_salary", value = np.round(predictions))

        val.set(data)
        
        return data

    # Descargar CSV con predicciones
    @session.download(filename = "predictions.csv")
    def download_predictions():
        yield val.get().to_csv(index=False)
    
    # Solo muestra la card de predicciones cuando se hace clic en el botón
    @output
    @render.ui
    def show_predictions():
        if input.calculate_button() > 0:  # El botón fue presionado
            return ui_card(
                "Predictions",
                ui.output_table("predictions"),
                ui.download_button("download_predictions", "Download CSV with predictions")
            )
        return ui.div()

# Crear la app
app = App(app_ui, server)
