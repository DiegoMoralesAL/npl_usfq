import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
import string
from nltk.corpus import stopwords
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
from moviepy.editor import AudioFileClip
from google.cloud import speech
from collections import Counter


# Función para limpiar las letras
def limpiar_texto(text):
    text = text.lower()  # Convertir a minúsculas
    text = ''.join([char for char in text if char not in string.punctuation])  # Quitar puntuación
    tokens = text.split()  # Tokenización
    text = ' '.join([word for word in tokens if word not in stopwords.words('english')])  # Quitar stopwords
    return text


def predecir_genero(model, nuevas_letras):
    tfidf = joblib.load('models/tfidf.pkl')
    nuevas_letras_limpias = limpiar_texto(nuevas_letras)
    letras_vectorizadas = tfidf.transform([nuevas_letras_limpias]).toarray()
    prediccion = model.predict(letras_vectorizadas)
    return prediccion[0]


def itags(url):
    yt = YouTube(url, on_progress_callback=on_progress)
    max_audio = 0
    audio_value = 0
    for audio_stream in yt.streams.filter(only_audio=True):
        abr = int(audio_stream.abr.replace('kbps', ''))
        if abr > max_audio:
            max_audio = abr
            audio_value = audio_stream.itag

    # Download the file
    downloaded_file = yt.streams.get_by_itag(audio_value).download()

    # Define the destination folder
    destination_folder = 'songs'

    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move the downloaded file to the 'songs' folder
    destination_path = os.path.join(destination_folder, os.path.basename(downloaded_file))
    os.rename(downloaded_file, destination_path)

    return destination_path


def extract_60_secs(audio_file):
    # Cargar el archivo de audio
    audio_clip = AudioFileClip(audio_file)

    # Duración total del archivo de audio
    audio_duration = audio_clip.duration

    # Calcular el punto de inicio y final para extraer 60 segundos desde la mitad
    start_time = (audio_duration / 2) - 30  # 30 segundos antes del punto medio
    end_time = start_time + 60  # 60 segundos desde el punto de inicio

    # Recortar el audio
    extracted_clip = audio_clip.subclip(start_time, end_time)

    # Generar el nombre dinámico del archivo de salida con el sufijo "-60secs"
    base_name, ext = os.path.splitext(audio_file)
    output_audio = base_name + '-60secs.mp3'

    # Guardar el nuevo archivo de audio en formato .mp3
    extracted_clip.write_audiofile(output_audio)

    # Cerrar los clips
    audio_clip.close()
    extracted_clip.close()

    print(f"Archivo guardado como: {output_audio}")
    return output_audio


def transcribe_audio(audio_file):
    client = speech.SpeechClient()

    # Cargar el archivo de audio
    with open(audio_file, "rb") as audio:
        content = audio.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="en-US"
    )

    # Realizar la transcripción
    response = client.recognize(config=config, audio=audio)

    results = []

    # Procesar y mostrar los resultados
    for result in response.results:
        results.append(result.alternatives[0].transcript)
        print(f"{result.alternatives[0].transcript}")

    return results