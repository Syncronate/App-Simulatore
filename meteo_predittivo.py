import streamlit as st
import pandas as pd
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

####################################################PARTE 1: DATA ENTRY (Streamlit) - Maschera per inserimento dati nel CSV####################################################
def data_entry_form(df, file_path="dataset_idrologico.csv"):
    """
    Streamlit form for data entry to add new data to the hydrological dataset.
    """
    st.header("Inserimento Nuovo Evento Idrologico")

    next_event_id = get_next_event_id(df)

    fields_info = {
        "data": {"label": "Data Evento", "type": "date", "default": datetime.now(), "streamlit_type": st.date_input}, # ADDED Data field
        "evento": {"label": "ID Evento", "type": "int", "default": next_event_id, "streamlit_type": st.number_input, "kwargs": {"format": "%d", "disabled": True}},
        "saturazione_terreno": {"label": "Saturazione Terreno (%)", "type": "float", "default": 35.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "ore_pioggia_totali": {"label": "Ore pioggia totali", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "cumulata_totale": {"label": "Cumulata Totale (mm)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "pioggia_gg_precedenti": {"label": "Pioggia gg Precedenti (mm)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "intensità_media": {"label": "Intensità Media (mm/h)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1008_inizio": {"label": "Idrometria 1008 Inizio (m)", "type": "float", "default": 0.5, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1112_inizio": {"label": "Idrometria 1112 Inizio (m)", "type": "float", "default": 0.8, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1112_max": {"label": "Idrometria 1112 Max (m)", "type": "float", "default": 0.8, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1283_inizio": {"label": "Idrometria 1283 Inizio (m)", "type": "float", "default": 1.2, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_3072_inizio": {"label": "Idrometria 3072 Inizio (m)", "type": "float", "default": 0.7, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}}
    }

    input_values = {}
    with st.form("data_entry_form"):
        col1, col2, col3 = st.columns(3)
        field_items = list(fields_info.items())
        for i, (field_name, field_info) in enumerate(field_items):
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                kwargs = {"label": field_info["label"], "value": field_info["default"]}
                kwargs.update(field_info.get("kwargs", {}))
                input_values[field_name] = field_info["streamlit_type"](**kwargs)

        col1, col2, col3 = st.columns(3)
        with col1:
            save_button = st.form_submit_button("Salva Dati", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Cancella Campi", use_container_width=True)
        with col3:
            view_button = st.form_submit_button("Visualizza Dataset", use_container_width=True)

        if save_button:
            save_data(input_values, df, file_path, fields_info)
        if clear_button:
            st.session_state['fields_cleared'] = True
        if view_button:
            st.session_state['dataset_view'] = True

    if st.session_state.get('dataset_view', False):
        view_dataset_streamlit(df)
        st.session_state['dataset_view'] = False # Reset flag after viewing

    if st.session_state.get('fields_cleared', False):
        clear_fields_streamlit(fields_info)
        st.session_state['fields_cleared'] = False # Reset flag after clearing

def get_next_event_id(df):
    if df.empty:
        return 1
    return int(df['evento'].max()) + 1

def validate_inputs(input_values, fields_info):
    valid_data = {}
    for field_name, field_info in fields_info.items():
        try:
            value = input_values[field_name]
            if field_info["type"] == "int":
                valid_data[field_name] = int(value)
            elif field_info["type"] == "float":
                valid_data[field_name] = float(value)
            elif field_info["type"] == "date": # Handle date type
                valid_data[field_name] = value.strftime('%Y-%m-%d') # Format date to string YYYY-MM-DD
            else:
                valid_data[field_name] = value
        except ValueError:
            st.error(f"Il valore per {field_info['label']} non è valido.")
            return None
    return valid_data

def save_data(input_values, df, file_path, fields_info):
    valid_data = validate_inputs(input_values, fields_info)
    if valid_data is None:
        return

    new_row = pd.DataFrame([valid_data])
    updated_df = pd.concat([df, new_row], ignore_index=True)

    try:
        updated_df.to_csv(file_path, index=False, sep='\t')
        st.success("Dati salvati con successo nel file CSV!")
        st.session_state['dataset'] = updated_df # Update session state dataset

        next_id = get_next_event_id(updated_df)
        fields_info["evento"]["default"] = next_id # Update default for next entry

    except Exception as e:
        st.error(f"Errore durante il salvataggio: {str(e)}")

def clear_fields_streamlit(fields_info):
    for field_name, field_info in fields_info.items():
        if field_name != "evento":
            fields_info[field_name]["default"] = field_info["default"] # Reset to original default
    st.session_state['fields_cleared'] = True # Set flag to re-render form with cleared fields

def view_dataset_streamlit(df):
    if not df.empty:
        # Aggiungi funzionalità di ordinamento e filtro
        st.subheader("Ultimi 10 eventi del dataset")
        st.dataframe(
            df.tail(10),
            column_config={
                "evento": st.column_config.NumberColumn(format="%d"),
                "idrometria_1112_max": st.column_config.NumberColumn(
                    "Idrometria 1112 Max (m)",
                    help="Valore massimo registrato",
                    format="%.2f",
                    step=0.01
                ),
                "data": st.column_config.DateColumn("Data Evento", format="YYYY-MM-DD") # Display Data column as Date
            },
            use_container_width=True,
            hide_index=True
        )

        # Visualizzazione statistica di base
        if len(df) > 5:  # Solo se abbiamo abbastanza dati
            st.subheader("Statistiche di base")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numero totale eventi", len(df))
                st.metric("Idrometria 1112 Max Media", f"{df['idrometria_1112_max'].mean():.2f} m")
            with col2:
                st.metric("Cumulata totale media", f"{df['cumulata_totale'].mean():.2f} mm")
                st.metric("Intensità media", f"{df['intensità_media'].mean():.2f} mm/h")
    else:
        st.info("Dataset vuoto.")
    st.session_state['dataset_view'] = True # Set flag to show dataset view

def prepare_initial_dataset(output_file="dataset_idrologico.csv"):
    """
    Carica il dataset dal CSV se esiste.
    Se il file non esiste, restituisce un DataFrame vuoto.
    """
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file, sep='\t')
            if 'data' in df.columns: # Try to parse 'data' column as datetime if exists
                try:
                    df['data'] = pd.to_datetime(df['data'])
                except (ValueError, TypeError):
                    st.warning("Impossibile convertire la colonna 'data' in formato data. Verificare il formato nel CSV.")

            st.success(f"File {output_file} caricato con successo.")
            return df
        except Exception as e:
            st.error(f"Errore nel caricamento del file esistente: {str(e)}")
            return pd.DataFrame() # Restituisci DataFrame vuoto in caso di errore
    else:
        st.info(f"File {output_file} non trovato. Inizializzando un dataset vuoto.")
        return pd.DataFrame(columns=['data', 'evento', 'saturazione_terreno', 'ore_pioggia_totali', 'cumulata_totale', 'pioggia_gg_precedenti', 'intensità_media', 'idrometria_1008_inizio', 'idrometria_1112_inizio', 'idrometria_1112_max', 'idrometria_1283_inizio', 'idrometria_3072_inizio']) # Return empty DataFrame with columns including 'data'


####################################################PARTE 2: SIMULAZIONE - Maschera per inserimento dati di simulazione (Streamlit)####################################################
def simulation_data_entry_form(feature_defaults, on_submit):
    """
    Streamlit form for simulation data entry.

    Args:
        feature_defaults: Dictionary with names and default values for each feature.
        on_submit: Callback function that receives the entered values when "Avvia Simulazione" is pressed.
    """
    st.header("Inserisci i dati per la simulazione")

    input_values = {}
    with st.form("simulation_form"):
        for field, default in feature_defaults.items():
            input_values[field] = st.number_input(field, value=default, format="%.2f")

        if st.form_submit_button("Avvia Simulazione"):
            try:
                values = [float(str(input_values[field]).replace(',', '.')) for field in feature_defaults] # Ensure comma is handled correctly
                on_submit(values)
            except ValueError:
                st.error("Verifica di aver inserito correttamente i valori numerici.")

####################################################Funzioni per salvare e caricare il modello####################################################
def salva_modello(model, scaler, file_path="model.pt", scaler_path="scaler.pkl"):
    """
    Salva il modello addestrato e lo scaler per un uso futuro.
    """
    # Salva il modello PyTorch
    torch.save(model.state_dict(), file_path)
    st.success(f"Modello salvato in {file_path}")

    # Salva lo scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    st.success(f"Scaler salvato in {scaler_path}")

def carica_modello(input_size, model_path="model.pt", scaler_path="scaler.pkl"):
    """
    Carica un modello precedentemente addestrato e lo scaler.
    """
    import os

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    try:
        # Inizializza un nuovo modello con la stessa architettura
        model = MeteoModel(input_size)
        # Carica i parametri del modello salvato
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load on CPU to avoid GPU issues
        model.eval()  # Imposta il modello in modalità valutazione

        # Carica lo scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        st.success("Modello e scaler caricati con successo!")
        return model, scaler
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello: {str(e)}")
        return None, None

####################################################PARTE 4: INTERFACCIA PER SIMULAZIONI MULTIPLE (Streamlit)####################################################
def multiple_simulations_interface(model, scaler, model_mae, features_cols, df):
    """
    Streamlit interface to run multiple simulations without retraining the model.
    """
    st.header("Simulazioni Multiple - Modello Addestrato")

    sim_defaults = {
        "saturazione_terreno": 44.0,
        "ore_pioggia_totali": 9.0,
        "cumulata_totale": 21.0,
        "pioggia_gg_precedenti": 0.0,
        "intensità_media": 2.3,
        "idrometria_1008_inizio": 0.54,
        "idrometria_1112_inizio": 1.18,
        "idrometria_1283_inizio": 1.18,
        "idrometria_3072_inizio": 0.89
    }

    # Aggiungi la possibilità di precompilare i campi da un evento esistente
    if not df.empty:
        st.subheader("Precompila campi da evento esistente")
        eventi_disponibili = ["Nessuno (usa valori predefiniti)"] # Initialize with default option
        for index, row in df.iterrows():
            event_label = f"Evento ID: {row['evento']}, Data: {row['data'].strftime('%Y-%m-%d') if isinstance(row['data'], pd.Timestamp) else row['data']}, Cumulata: {row['cumulata_totale']:.2f}, Idro Max 1112: {row['idrometria_1112_max']:.2f}" # Formatted label
            eventi_disponibili.append(event_label)

        selected_event_label = st.selectbox("Seleziona evento", eventi_disponibili)

        if selected_event_label != "Nessuno (usa valori predefiniti)":
            selected_event_id = int(selected_event_label.split("Evento ID: ")[1].split(",")[0]) # Extract event ID from label
            event_data = df[df['evento'] == selected_event_id].iloc[0]
            for field in sim_defaults.keys():
                if field in event_data:
                    sim_defaults[field] = float(event_data[field])
            st.success(f"Campi precompilati con evento {selected_event_id}")

    # Container per il modulo di input e il grafico
    col1, col2 = st.columns([1, 2])

    with col1:
        # Modulo di input più compatto
        with st.form("simulation_parameters_form"):
            st.subheader("Parametri simulazione")
            input_values = {}

            for field, default in sim_defaults.items():
                input_values[field] = st.number_input(
                    field,
                    value=default,
                    format="%.2f",
                    step=0.01,
                    label_visibility="visible" # or "collapsed" to hide labels above inputs
                )

            simulate_button = st.form_submit_button("Esegui Simulazione", use_container_width=True)
            clear_button = st.form_submit_button("Pulisci Campi", use_container_width=True)

            if simulate_button:
                st.session_state['simulation_run'] = True
                st.session_state['simulation_input_values'] = input_values

            if clear_button:
                st.session_state['simulation_fields_cleared'] = True

    # Visualizzazione risultati simulazione
    if st.session_state.get('simulation_run', False):
        input_values = st.session_state.get('simulation_input_values', sim_defaults)
        values = get_simulation_input_values(input_values)

        if values is not None:
            # Esegui la previsione
            predizione_valore, range_previsione, livelli_idrometrici = simula_previsione(
                model, scaler, values, model_mae
            )

            # Memorizza i risultati della previsione nella session state
            st.session_state['predizione_valore'] = predizione_valore
            st.session_state['range_previsione'] = range_previsione
            st.session_state['valori_input'] = values

            # Genera e visualizza sia il grafico Plotly che Matplotlib
            with col2:
                # Visualizzazione interattiva Plotly
                fig_plotly = visualizza_previsione_plotly(predizione_valore, range_previsione, values)
                st.plotly_chart(fig_plotly, use_container_width=True)

                # Checkbox to toggle Matplotlib visualization
                show_matplotlib = st.checkbox("Visualizzazione Tradizionale (Matplotlib)", value=False)
                if show_matplotlib:
                    fig_mpl = visualizza_previsione_idrometrica(predizione_valore, range_previsione, livelli_idrometrici, values)
                    st.pyplot(fig_mpl)

                # Mostra anche i dati numerici e lo stato
                st.subheader("Risultati Simulazione")
                col_ris1, col_ris2 = st.columns(2)
                with col_ris1:
                    st.metric("Previsione Idrometria 1112 Max", f"{predizione_valore:.2f} m ± {model_mae:.2f} m")
                with col_ris2:
                    # Stato della previsione con colore adeguato
                    livello_attenzione, livello_preallarme, livello_allarme = 1.5, 2.0, 3.0
                    if predizione_valore < livello_attenzione:
                        st.success("Stato: NORMALE")
                    elif predizione_valore < livello_preallarme:
                        st.warning("Stato: ATTENZIONE")
                    elif predizione_valore < livello_allarme:
                        st.warning("Stato: PREALLARME")
                    else:
                        st.error("Stato: ALLARME")


        st.session_state['simulation_run'] = False
        st.session_state['simulation_input_values'] = None

    # Se è la prima visualizzazione, mostra un placeholder
    elif col2.container():
        with col2:
            st.info("Inserisci i parametri e clicca 'Esegui Simulazione' per vedere i risultati.")

    if st.session_state.get('simulation_fields_cleared', False):
        clear_simulation_fields_streamlit(sim_defaults)
        st.session_state['simulation_fields_cleared'] = False


def clear_simulation_fields_streamlit(sim_defaults):
    """Pulisce i campi di input simulazione e li resetta ai valori predefiniti."""
    for field, default in sim_defaults.items():
        sim_defaults[field] = default # No need to set value directly, Streamlit form will handle on re-run
    st.session_state['simulation_fields_cleared'] = True # Set flag to re-render form with cleared fields


def get_simulation_input_values(input_values):
    """Ottiene i valori di input dai campi simulazione."""
    try:
        values = []
        for field in input_values.keys():
            value = float(str(input_values[field]).replace(',', '.'))
            values.append(value)
        return values
    except ValueError:
        st.error("Inserisci valori numerici validi in tutti i campi di simulazione!")
        return None

def run_simulation_streamlit(model, scaler, model_mae, input_values):
    """Esegui la simulazione con i parametri correnti."""
    values = get_simulation_input_values(input_values)
    if values is None:
        return

    # Esegui la previsione
    predizione_valore, range_previsione, livelli_idrometrici = simula_previsione(
        model, scaler, values, model_mae
    )

    # Crea e visualizza il grafico
    fig = visualizza_previsione_idrometrica(predizione_valore, range_previsione, livelli_idrometrici, values)
    st.pyplot(fig)

####################################################
# Funzione di simulazione della previsione
####################################################
def simula_previsione(modello, scaler, input_features_values, model_mae):
    modello.eval()
    with torch.no_grad():
        nuovi_dati_input_np = np.array(input_features_values).reshape(1, -1)
        nuovi_dati_input_scaled = scaler.transform(nuovi_dati_input_np)
        nuovi_dati_input_tensor = torch.tensor(nuovi_dati_input_scaled, dtype=torch.float32)
        predizione_tensor = modello(nuovi_dati_input_tensor)
        predizione_valore = predizione_tensor.item()

    range_previsione = (predizione_valore - model_mae, predizione_valore + model_mae)
    livelli_idrometrici_input = {"Idrometria 1112 Max (Predetta)": predizione_valore}
    return predizione_valore, range_previsione, livelli_idrometrici_input

####################################################
# NUOVA FUNZIONE: Visualizzazione Plotly interattiva
####################################################
def visualizza_previsione_plotly(previsione, range_previsione, input_values):
    """
    Crea una visualizzazione interattiva della previsione usando Plotly.
    """
    # Livelli di riferimento
    livello_attenzione = 1.5
    livello_preallarme = 2.0
    livello_allarme = 3.0

    # Determina la categoria della previsione
    if previsione < livello_attenzione:
        categoria = "NORMALE"
        colore_categoria = "#00cc00"
    elif previsione < livello_preallarme:
        categoria = "ATTENZIONE"
        colore_categoria = "#ffcc00"
    elif previsione < livello_allarme:
        categoria = "PREALLARME"
        colore_categoria = "#ff9900"
    else:
        categoria = "ALLARME"
        colore_categoria = "#ff3300"

    # Crea il grafico base
    fig = go.Figure()

    # Aggiungi aree colorate per le zone
    fig.add_shape(
        type="rect",
        x0=0, x1=livello_attenzione,
        y0=0, y1=1,
        fillcolor="#00cc00",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=livello_attenzione, x1=livello_preallarme,
        y0=0, y1=1,
        fillcolor="#ffcc00",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=livello_preallarme, x1=livello_allarme,
        y0=0, y1=1,
        fillcolor="#ff9900",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=livello_allarme, x1=4,
        y0=0, y1=1,
        fillcolor="#ff3300",
        opacity=0.1,
        layer="below",
        line_width=0,
    )

    # Aggiungi il range di incertezza
    fig.add_shape(
        type="rect",
        x0=range_previsione[0], x1=range_previsione[1],
        y0=0, y1=1,
        fillcolor="#4287f5",
        opacity=0.3,
        line_width=0,
    )

    # Aggiungi linee per i livelli di soglia
    fig.add_shape(
        type="line",
        x0=livello_attenzione, x1=livello_attenzione,
        y0=0, y1=1,
        line=dict(color="#ffcc00", width=2, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=livello_preallarme, x1=livello_preallarme,
        y0=0, y1=1,
        line=dict(color="#ff9900", width=2, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=livello_allarme, x1=livello_allarme,
        y0=0, y1=1,
        line=dict(color="#ff3300", width=2, dash="dash"),
    )

    # Aggiungi linea per la previsione
    fig.add_shape(
        type="line",
        x0=previsione, x1=previsione,
        y0=0, y1=1,
        line=dict(color="#00ffcc", width=3),
    )

    # Aggiungi etichette per le zone
    fig.add_annotation(
        x=livello_attenzione/2, y=0.98,
        text="NORMALE",
        showarrow=False,
        font=dict(color="#00cc00", size=14)
    )
    fig.add_annotation(
        x=(livello_attenzione+livello_preallarme)/2, y=0.98,
        text="ATTENZIONE",
        showarrow=False,
        font=dict(color="#ffcc00", size=14)
    )
    fig.add_annotation(
        x=(livello_preallarme+livello_allarme)/2, y=0.98,
        text="PREALLARME",
        showarrow=False,
        font=dict(color="#ff9900", size=14)
    )
    fig.add_annotation(
        x=(livello_allarme+4)/2, y=0.98,
        text="ALLARME",
        showarrow=False,
        font=dict(color="#ff3300", size=14)
    )

    # Aggiungi una riga invisibile per attivare il layout
    fig.add_trace(go.Scatter(
        x=[0, 4],
        y=[0.5, 0.5],
        mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False
    ))

    # Aggiungi testo informativo sui parametri di input
    info_text = (
        f"<b>Parametri di input:</b><br>" +
        f"Saturazione terreno: {input_values[0]:.1f}%<br>" +
        f"Ore pioggia: {input_values[1]:.1f}<br>" +
        f"Cumulata: {input_values[2]:.1f} mm<br>" +
        f"Pioggia gg precedenti: {input_values[3]:.1f} mm<br>" +
        f"Intensità media: {input_values[4]:.1f} mm/h<br>" +
        f"Idrometria 1112 iniziale: {input_values[6]:.2f} m"
    )

    # Aggiungi legenda per il livello di previsione
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="#00ffcc", size=15),
        name=f"Previsione: {previsione:.2f} m"
    ))

    # Aggiungi legenda per l'incertezza
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="#4287f5", size=15, opacity=0.5),
        name=f"Incertezza: ±{(range_previsione[1]-range_previsione[0])/2:.2f} m"
    ))

    # Aggiungi legenda per le soglie
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#ffcc00", width=2, dash="dash"),
        name="Livello attenzione: 1.50 m"
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#ff9900", width=2, dash="dash"),
        name="Livello preallarme: 2.00 m"
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#ff3300", width=2, dash="dash"),
        name="Livello allarme: 3.00 m"
    ))

    # Configura il layout
    fig.update_layout(
        title={
            'text': f"Previsione Idrometria 1112 Max - {categoria}",
            'font': {'size': 24, 'color': colore_categoria},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Livello idrometrico (m)",
        xaxis=dict(
            range=[0, 4],
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0, 1]
        ),
        annotations=[
            dict(
                x=0.02,
                y=0.15,
                xref="paper",
                yref="paper",
                text=info_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.5)",
                borderwidth=1,
                borderpad=10,
                opacity=0.8
            )
        ],
        template="plotly_dark",
        margin=dict(l=20, r=20, t=100, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

####################################################
# Funzione per la visualizzazione grafica della previsione (Matplotlib - Tradizionale)
####################################################
def visualizza_previsione_idrometrica(previsione, range_previsione, livelli_idrometrici, input_values):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    livello_attenzione = 1.5
    livello_preallarme = 2.0
    livello_allarme = 3.0

    ax.axvspan(range_previsione[0], range_previsione[1], alpha=0.3, color='#4287f5',
               label=f'Range incertezza (±{(range_previsione[1]-range_previsione[0])/2:.2f})')

    ax.axvline(x=livello_attenzione, color='#ffcc00', linestyle='--', alpha=0.7, label='Livello attenzione')
    ax.axvline(x=livello_preallarme, color='#ff9900', linestyle='--', alpha=0.7, label='Livello preallarme')
    ax.axvline(x=livello_allarme, color='#ff3300', linestyle='--', alpha=0.7, label='Livello allarme')

    ax.axvline(x=previsione, color='#00ffcc', linewidth=2.5, label=f'Previsione: {previsione:.2f}')

    ax.axvspan(0, livello_attenzione, alpha=0.1, color='#00cc00')
    ax.axvspan(livello_attenzione, livello_preallarme, alpha=0.1, color='#ffcc00')
    ax.axvspan(livello_preallarme, livello_allarme, alpha=0.1, color='#ff9900')
    ax.axvspan(livello_allarme, 4, alpha=0.1, color='#ff3300')

    plt.text(livello_attenzione/2, 0.02, 'NORMALE', color='#00cc00', ha='center')
    plt.text((livello_attenzione+livello_preallarme)/2, 0.02, 'ATTENZIONE', color='#ffcc00', ha='center')
    plt.text((livello_preallarme+livello_allarme)/2, 0.02, 'PREALLARME', color='#ff9900', ha='center')
    plt.text((livello_allarme+4)/2, 0.02, 'ALLARME', color='#ff3300', ha='center')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    plt.title('Previsione Idrometria 1112 Max', fontsize=16, pad=20)
    plt.xlabel('Livello idrometrico (m)', fontsize=12, labelpad=10)

    info_text = (
        f"Saturazione terreno: {input_values[0]:.1f}%\n"
        f"Ore pioggia: {input_values[1]:.1f}\n"
        f"Cumulata: {input_values[2]:.1f} mm\n"
        f"Intensità media: {input_values[4]:.1f} mm/h\n"
        f"Idrometria 1112 iniziale: {input_values[6]:.2f} m"
    )

    plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

    livello_attenzione, livello_preallarme, livello_allarme = 1.5, 2.0, 3.0
    if previsione < livello_attenzione:
        zona = "NORMALE"
        prob_text = "Probabilità di superamento\ndel livello di attenzione: BASSA"
        prob_color = '#00cc00'
    elif previsione < livello_preallarme:
        zona = "ATTENZIONE"
        prob_text = "Probabilità di superamento\ndel livello di preallarme: MEDIA"
        prob_color = '#ffcc00'
    elif previsione < livello_allarme:
        zona = "PREALLARME"
        prob_text = "Probabilità di superamento\ndel livello di allarme: ALTA"
        prob_color = '#ff9900'
    else:
        zona = "ALLARME"
        prob_text = "Livello di allarme\nsuperato"
        prob_color = '#ff3300'

    plt.figtext(0.85, 0.2, prob_text, fontsize=10, color=prob_color,
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

    plt.figtext(0.5, 0.9, f'Classificazione: {zona}', fontsize=18, color=prob_color,
                ha='center', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))

    ax.legend(loc='upper right', framealpha=0.8)

    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    return fig

####################################################
# Definizione del Modello PyTorch (No change needed)
####################################################
class MeteoModel(nn.Module):
    def __init__(self, input_size):
        super(MeteoModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

####################################################MAIN: Esecuzione sequenziale: Data Entry -> Inserimento dati simulazione -> Simulazione (Streamlit Main)####################################################
if __name__ == "__main__":
    st.title("Dashboard Idrologico")

    # Initialize session state for dataset and flags
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = prepare_initial_dataset()
    if 'dataset_view' not in st.session_state:
        st.session_state['dataset_view'] = False
    if 'fields_cleared' not in st.session_state:
        st.session_state['fields_cleared'] = False
    if 'simulation_run' not in st.session_state:
        st.session_state['simulation_run'] = False
    if 'simulation_fields_cleared' not in st.session_state:
        st.session_state['simulation_fields_cleared'] = False
    if 'simulation_input_values' not in st.session_state:
        st.session_state['simulation_input_values'] = None
    if 'simulation_plot' not in st.session_state:
        st.session_state['simulation_plot'] = None
    if 'predizione_valore' not in st.session_state:
        st.session_state['predizione_valore'] = None
    if 'range_previsione' not in st.session_state:
        st.session_state['range_previsione'] = None
    if 'valori_input' not in st.session_state:
        st.session_state['valori_input'] = None
    if 'retrain_model' not in st.session_state:
        st.session_state['retrain_model'] = False # Inizializza la variabile per il riallenamento


    dataset_csv = "dataset_idrologico.csv"
    df = st.session_state['dataset']

    # PART 1: Data Entry Form
    with st.expander("Inserimento Dati", expanded=False):
        data_entry_form(df, file_path=dataset_csv)
    df = pd.read_csv(dataset_csv, sep='\t', dtype=str) # Reload dataset after potential save operation
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    for col in df.columns:
        if col != 'evento' and col != 'data': # Exclude 'data' column from numeric conversion
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    df['evento'] = df['evento'].astype(int)
    if 'data' in df.columns: # Convert 'data' column to datetime after reload
        try:
            df['data'] = pd.to_datetime(df['data'])
        except (ValueError, TypeError):
            st.warning("Impossibile convertire la colonna 'data' in formato data. Verificare il formato nel CSV.")
    st.session_state['dataset'] = df # Update dataset in session state


    # Data preparation and model loading/training (same as before, but in Streamlit context)
    features_cols = [
        'saturazione_terreno',
        'ore_pioggia_totali',
        'cumulata_totale',
        'pioggia_gg_precedenti',
        'intensità_media',
        'idrometria_1008_inizio',
        'idrometria_1112_inizio',
        'idrometria_1283_inizio',
        'idrometria_3072_inizio'
    ]
    target_col = 'idrometria_1112_max'

    X = df[features_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_size = X_train_scaled.shape[1]
    model, loaded_scaler = carica_modello(input_size)

    if model is None or loaded_scaler is None or st.session_state.get('retrain_model'): # Condizione MODIFICATA
        if st.session_state.get('retrain_model'):
            st.info("Riallenamento del modello richiesto dall'utente...")
        else:
            st.info("Nessun modello esistente trovato. Addestramento di un nuovo modello...")

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        model = MeteoModel(input_size)

        learning_rate = 0.001
        num_epochs = 2000
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 100 == 0:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') # st.write for Streamlit display

        st.success('Training finito!')

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            st.write(f'Loss sul Test Set: {test_loss.item():.4f}')

        y_pred_test = test_outputs.numpy()
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        st.write(f'MAE sul Test Set: {mae:.4f}')
        st.write(f'R^2 sul Test Set: {r2:.4f}')

        salva_modello(model, scaler)

        if st.session_state.get('retrain_model'):
            st.session_state['retrain_model'] = False # Resetta la variabile di sessione
            st.sidebar.success("Modello riallenato con successo!") # Feedback di successo nella sidebar
    else:
        scaler = loaded_scaler

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            criterion = nn.MSELoss()
            test_loss = criterion(test_outputs, y_test_tensor)
            st.write(f'Loss sul Test Set del modello caricato: {test_loss.item():.4f}')

        y_pred_test = test_outputs.numpy()
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        st.write(f'MAE sul Test Set: {mae:.4f}')
        st.write(f'R^2 sul Test Set: {r2:.4f}')

    # PART 4: Multiple Simulations Interface
    with st.expander("Simulazioni Multiple", expanded=True):
        multiple_simulations_interface(model, scaler, mae, features_cols, df)

    st.sidebar.header("Informazioni")
    st.sidebar.info("Questa dashboard permette l'inserimento di dati idrologici, la visualizzazione del dataset, e l'esecuzione di simulazioni previsionali sul livello idrometrico. La parte grafica della simulazione è integrata direttamente nella dashboard sotto il form di simulazione.")

    # Aggiungi il pulsante per riallenare il modello (AGGIUNTO)
    if st.sidebar.button("Riallena Modello"):
        st.session_state['retrain_model'] = True
        st.sidebar.info("Riallenamento del modello in corso...") # Feedback per l'utente
