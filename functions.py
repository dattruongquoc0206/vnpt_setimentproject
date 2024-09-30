import os
import librosa
import numpy as np
import sqlite3
import time

def get_size_kb(file_path):
    return os.path.getsize(file_path)/1024

def calculate_silence_ratio(file_path, silence_threshold=1e-4, frame_length=2048, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Calculate the total duration of the audio in seconds
    total_duration = len(y) / sr

    # Calculate the short-term energy for each frame
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])

    # Define silence as frames where energy is below the threshold
    silent_frames = np.sum(energy < silence_threshold)

    # Calculate the duration of silent frames in seconds
    silent_duration = (silent_frames * hop_length) / sr

    # Calculate the silence ratio
    silence_ratio = silent_duration / total_duration

    return silence_ratio

def get_wav_informations(file_path):
    # name = os.path.basename(file_path)
    size = get_size_kb(file_path)
    wav, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=wav, sr=sr)
    channel = len(wav.shape)
    silence_ratio = calculate_silence_ratio(file_path)
    size = get_size_kb(file_path)
    # creation_time = os.path.getctime(file_path)
    return size, sr, channel, duration, silence_ratio, wav