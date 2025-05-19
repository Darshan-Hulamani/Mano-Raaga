import streamlit as st 
import mysql.connector 
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import re
import base64
import hashlib
import streamlit.components.v1 as components
from pathlib import Path
import threading
from PIL import Image
from testdata2 import recognize

# For music.py
import asyncio
import nest_asyncio 
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase 
import av 
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import webbrowser 
import os
import pygame
from pygame import mixer 
import time
from mutagen.mp3 import MP3 
import tempfile
import shutil

import warnings
from absl import logging
import os
import tensorflow as tf

# -------------------------------
# Config & Init
# -------------------------------
st.set_page_config(page_title="Mano-Raaga", page_icon="images/logo3(1).ico", layout="centered", initial_sidebar_state="auto", menu_items=None)
pygame.mixer.init()


# -------------------------------
# Page Design and Setup
# -------------------------------
# Custom CSS
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Playing default background musicand stopping after capturing the emotion
def play_background_music():
    if not st.session_state.background_music_playing and not st.session_state.get('user'):
        try:
            mixer.music.load("images/default_background.mp3")
            mixer.music.play(-1)  # -1 makes it loop indefinitely
            # mixer.music.set_volume(0.5)  # Lower volume for background music
            st.session_state.background_music_playing = True
        except:
            pass  # Silently fail if music file not found

def stop_background_music():
    if st.session_state.background_music_playing:
        mixer.music.stop()
        st.session_state.background_music_playing = False

def display_logo():
    """Display application logo in sidebar."""
    try:
        image_path = "images/logo3.png"
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode()

        st.sidebar.markdown(
            f"""
            <div style="text-align: center;">
            <img id="img" src="data:image/jpg;base64,{encoded_image}" alt="Image Not Found" width="100px" height="100px">
            </div>
            """, unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Couldn't load image: {str(e)}")

def display_sidebar_header():
    """Display sidebar header."""
    st.sidebar.markdown(
        """
        <head>
            <link href='https://fonts.googleapis.com/css?family=Bubblegum Sans' rel='stylesheet'>
            <style>
                #ftstyl{
                    text-align: center;
                    font-family: 'Bubblegum Sans';
                }
            </style>
        </head>
        <h3 id="ftstyl" style="padding-bottom: 1px;">Mano-Raaga</h3>
        <h4 id="ftstyl" style="font-weight: lighter;">🌊 Not just surfing waves — we surf moods too! 🎵✨</h4>
        """, unsafe_allow_html=True)
    

def set_background_image(image_path):
    """Set the background image for the app."""
    try:
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpg;base64,{encoded_image}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        # st.error(f"Couldn't load background image: {str(e)}")
        pass

# -------------------------------
# Database Connection
# -------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        port=st.secrets["mysql"].get("port", 3306)
    )

# -------------------------------
# Initialize Session State
# -------------------------------
def initialize_session():
    defaults = {
        'user': None,
        'emotion': None,
        'lang': None,
        'current_page': "None",
        'current_song_index': 0,
        'is_playing': False,
        'paused': False,
        'volume': 0.5,
        'track_position': 0,
        'start_time': time.time(),
        'selected_playlist': "None",
        'background_music_playing': False,
        'capture': False,
        'run': "true",
        'temp_dir': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------
# Music Player Helpers
# -------------------------------
def play_song(song_path, start_position=0):
    """Play the song from a specific position."""
    try:
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play(start=start_position)
        pygame.mixer.music.set_volume(st.session_state.volume)
        st.session_state.is_playing = True
        st.session_state.paused = False
        st.session_state.track_position = start_position
        st.session_state.start_time = time.time() - start_position  # Adjust start time
    except:
        pass

def pause_song():
    """Pause the song and store the track position."""
    pygame.mixer.music.pause()
    st.session_state.is_playing = False
    st.session_state.paused = True
    st.session_state.track_position = int(time.time() - st.session_state.start_time)  # Save current position

def resume_song():
    """Resume the song from the paused position."""
    pygame.mixer.music.unpause()
    st.session_state.is_playing = True
    st.session_state.paused = False
    st.session_state.start_time = time.time() - st.session_state.track_position  # Adjust start time

def seek_song(position, song_path):
    """Seek to a new position in the song."""
    pygame.mixer.music.stop()
    play_song(song_path, start_position=position)
    st.session_state.track_position = position
    st.session_state.start_time = time.time() - position  # Adjust start time

def get_song_length(song_path):
    """Get the length of the song in seconds."""
    try:
        audio = MP3(song_path)
        return int(audio.info.length)
    except:
        return 1  # Default to 1 second if error

def get_current_position():
    """Get the current playback position."""
    if st.session_state.is_playing:
        return int(time.time() - st.session_state.start_time)  # Compute actual position
    return int(st.session_state.track_position)  # Ensure it's always an integer

def refresh():
    # Auto-refresh every second
    time.sleep(5)
    st.rerun()

# -------------------------------
# Song Folder Setup
# -------------------------------
def create_emotion_playlist():
    emotion = st.session_state['emotion']
    language = st.session_state['lang']
    if not (emotion and language and language != "Select Language"):
        return

    current_directory = os.getcwd()
    temp_dir = tempfile.mkdtemp(dir = current_directory)
    st.session_state['temp_dir'] = temp_dir
    st.session_state['selected_playlist'] = temp_dir

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT music_id FROM music_mapping WHERE language_id = (SELECT language_id FROM language WHERE language_name = %s) AND emotion_id = (SELECT emotion_id FROM emotion WHERE emotion = %s)", (str(language), str(emotion),))
        music_ids = cursor.fetchall()

        for music_id_tuple in music_ids:
            music_id = music_id_tuple[0]
            cursor.execute("SELECT music_name FROM music WHERE music_id = %s", (music_id,))
            music_name_row = cursor.fetchone()
            if music_name_row:
                music_name = music_name_row[0] + ".mp3"
                source_path = os.path.join('songs',music_name)
                temp_file_path = os.path.join(temp_dir, music_name)
                shutil.copy(source_path, temp_file_path)

    except Exception as e:
        st.error(f"Playlist creation error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

if not st.session_state.get('selected_playlist') or st.session_state.get('selected_playlist') == "None":
    song_folder = "songs"
else:
    song_folder = st.session_state.selected_playlist

filtered_mp3_files = [file for file in os.listdir(song_folder) if file.endswith(".mp3")]

def get_filtered_songs():
    path = st.session_state.get('selected_playlist', "songs")
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path) if f.endswith(".mp3")]

def cleanup_temp_files():
    """Clean up all temporary files and directories created during the session."""
    try:
        # Don't cleanup if music is currently playing from temp directory
        path = '.'
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if folder.startswith('tmp') and os.path.isdir(folder_path):
                if folder:
                    print(f"Found {folder}. Deleting folder...")
                    shutil.rmtree(folder_path)  # This will delete the whole folder
                    print(f"Folder {folder} has been deleted.")

        # Remove captured images directory
        if os.path.exists("captured_images"):
            print("Found captured_images. Deleting folder...")
            shutil.rmtree("captured_images")
            
        # Remove emotion.npy file if exists
        if os.path.exists("emotion.npy"):
            print("Found emotion.npy. Deleting file...")
            os.remove("emotion.npy")
            
        # Remove __pycache__ file if exists
        if os.path.exists("__pycache__"):
            print("Found __pycache__. Deleting folder...")
            shutil.rmtree("__pycache__")

        return True

    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")
        return False

def initial_cleanup_temp_files():
    """Clean up all temporary files and directories created during the session whenever new session is started."""
    try:
        if st.session_state.temp_dir:
            tmp_dir = st.session_state.temp_dir
            path = '.'
            for folder in os.listdir(path):
                folder_path = os.path.join(path, folder)
                if folder.startswith('tmp') and os.path.isdir(folder_path):
                    if folder != tmp_dir:
                        st.write(f"Found {folder}. Deleting folder...")
                        shutil.rmtree(folder_path)  # This will delete the whole folder
                        print(f"Folder {folder} has been deleted.")
        
        # Remove captured images directory
        if os.path.exists("captured_images"):
            print("Removed captured images directory...")
            shutil.rmtree("captured_images")
            
        # Remove emotion.npy file if exists
        if os.path.exists("emotion.npy"):
            print("Removed emotion.npy file...")
            os.remove("emotion.npy")
            
        return True

    except Exception as e:
        pass

# -------------------------------
# Password Validation Function
# -------------------------------
def validate_password(password):
    """Validate password strength."""
    if len(password) < 8:
        return "❌ Must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "❌ Include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "❌ Include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "❌ Include at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "❌ Include at least one special character."
    return "✅ Strong password!"

# -------------------------------
# Stubbed Pages
# -------------------------------
def login():
    """Handle user login."""
    st.title("Login")
    st.toast("New to Mano-Raag? Register it fastttt..!", icon="🏃🏻‍➡️")

    with st.form("login_form"):
        username = st.text_input("Email", placeholder="Enter your Email")
        password = st.text_input("Password", type="password", placeholder="Enter your Password")
        
        if st.form_submit_button("Login", icon="🔑"):
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                sha = hashlib.sha256(password.encode('utf-8')).hexdigest()
                cursor.execute(
                    "SELECT * FROM passwords_backup WHERE user_id = (SELECT id FROM users WHERE email = %s);",
                    (username,)
                )
                user = cursor.fetchone()

                if user and sha == user[2]:
                    st.session_state['user'] = username
                    st.success("Login successful!")
                    #stop_background_music()
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            except mysql.connector.Error as err:
                st.error(f"Database error: {err}")
            finally:
                cursor.close()
                conn.close()

def register():
    """Handle user registration."""
    st.title("Register")
    st.toast("Already a user? Then enjoy 'Mano-Raag' by Logging In..! 🎵", icon="⁉️")

    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Enter Password", type="password", 
                               placeholder="Password must contain min 8 chars with a Capital, Small, Number & Special Char")
        
        if password:
            message = validate_password(password)
            (st.success if "✅" in message else st.error)(message)

        confirm_password = st.text_input("Confirm Password", type="password", 
                                        placeholder="Re-enter the Password")
        
        if confirm_password and password != confirm_password:
            st.error("Password Doesn't Match")

        # Restricting for less than 3 years old for data safety
        cur_date = datetime.now().date()
        max_date = cur_date - timedelta(days=3*365)
        dob = st.date_input("Date of Birth", max_value=max_date)
        gender = st.selectbox("Gender", ["Male", "Female", "Others"])

        if st.form_submit_button("Register", icon="📝") and password == confirm_password:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
                email_exists = cursor.fetchone()[0]

                if email_exists:
                    st.error("⚠️ Email already exists. Try another one.")
                    return
                
                hashed_pwd = hashlib.sha256(password.encode('utf-8')).hexdigest()
                
                # Insert user data
                cursor.execute(
                    "INSERT INTO users (username, email, dob, gender) VALUES (%s, %s, %s, %s)",
                    (username, email, dob, gender)
                )
                conn.commit()

                # Get user ID
                cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
                user_id = cursor.fetchone()

                if user_id:
                    cursor.execute(
                        "INSERT INTO passwords_backup(user_id, hash_pwd) VALUES (%s, %s)",
                        (user_id[0], hashed_pwd)
                    )
                    conn.commit()
                
                st.snow()
                st.toast("Now enjoy the unimaginale experince in Mano-Raag's Music World..!🎵😊")
                st.success("🎉Registration successful! Please log in.")

            except mysql.connector.Error as err:
                st.error(f"Database error: {err}")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()

def forgot_password():
    """Handle password reset."""
    st.title("Forgot Password")

    with st.form("forgot_password_form"):
        email = st.text_input("Email", placeholder="Enter the correct email")
        dob = st.date_input("Date of Birth")
        password = st.text_input("Enter Password", type="password", 
                               placeholder="Password must contain min 8 chars with a Capital, Small, Number & Special Char")
        
        if password:
            message = validate_password(password)
            (st.success if "✅" in message else st.error)(message)

        confirm_password = st.text_input("Confirm Password", type="password", 
                                       placeholder="Re-enter the Password")
        
        if confirm_password and password != confirm_password:
            st.error("Password Doesn't Match")

        if st.form_submit_button("Reset Password"):
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                dob_str = dob.strftime('%Y-%m-%d')
                cursor.execute(
                    "SELECT * FROM users WHERE email = %s AND dob = %s", 
                    (email, dob_str)
                )
                user = cursor.fetchone()

                if user:
                    uid = user[0]
                    cursor.execute(
                        "SELECT hash_pwd FROM passwords_backup WHERE user_id = %s", 
                        (uid,)
                    )
                    pwd = cursor.fetchone()

                    sha = hashlib.sha256(password.encode('utf-8')).hexdigest()
                    
                    if sha == pwd[0]:
                        st.error("Password Cannot be same as previous")
                    else:
                        cursor.execute(
                            "UPDATE passwords_backup SET hash_pwd = %s WHERE user_id = %s", 
                            (sha, uid)
                        )
                        conn.commit()
                        st.success("Password Changed Successfully!")
                else:
                    st.warning("Your Email or DOB doesn't match in our database. Please try again or register.")
            
            except mysql.connector.Error as err:
                st.error(f"Database error: {err}")
            finally:
                cursor.close()
                conn.close()

def capture():
    """Capture user's emotion through camera."""
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    # Set the directory to save images
    SAVE_DIR = "captured_images"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Apply nested asyncio fix
    nest_asyncio.apply()

    # Ensure there's an active event loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if st.session_state['user']:
        mail = st.session_state['user']
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE email = %s", (mail,))
            usr = cursor.fetchone()
            st.title(f"Hi {usr[0]}, This is Music Recommending Software Application")
        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
        finally:
            cursor.close()
            conn.close()
                
        st.write("Please, Let me Capture your Emotion..!")
        st.write("Please give me Camera Access.")

        emotion = ""
        if not emotion:
            st.session_state['run'] = "true"
        else:
            st.session_state['run'] = "false"

        lang = st.selectbox("Languages", ["Select Language", "Kannada", "Hindi", "English", "Tamil", "Telugu"])
        st.session_state['lang'] = lang
        
        if st.button("Start Capturing", icon="📸"):
            st.session_state['capture'] = "true"

        if (lang != "Select Language") and (st.session_state['run'] == "true") and (st.session_state['capture'] == "true"):
            img_file_buffer = st.camera_input("Take a picture")

            if img_file_buffer is not None:
                try:
                    image = Image.open(img_file_buffer)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"photo_{timestamp}.png"
                    filepath = os.path.join(SAVE_DIR, filename)
                    image.save(filepath)
                    st.success(f"✅ Image saved to {filepath}")
                    st.session_state['emotion'] = recognize(filename)
                    st.write(f"{st.session_state['emotion']}")
                    if st.session_state['emotion'] and st.session_state['lang'] != "Select Language":
                        create_emotion_playlist()
                        
                except Exception as e:
                    st.warning("Emotion not yet Captured please Clear Photo and Re-capture it...")


        btn = st.button("Let's Gooo..!")

        if btn:
            if not emotion:
                st.warning("Please let me capture your emotion first")
                st.session_state['run'] = "true"
            else:
                st.write("You can go to Music Page now...") 
                st.session_state['run'] = "false"
                st.session_state['capture'] = None
    else:
        st.warning("Please login first.")

def dashboard():
    """Application dashboard page."""
    set_background_image("images/bg6_music.jpg")
    if st.session_state['user']:
        mail = st.session_state['user']
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE email = %s", (mail,))
            usr = cursor.fetchone()
            st.subheader(f"Hi {usr[0]}!")
        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
        finally:
            cursor.close()
            conn.close()
    
    st.title("Welcome to Mano-Raaga..! ")
    st.header("🌊 Not just surfing waves — we surf moods too! 🎵✨")
    
    # About Us Content
    st.header("About Us")

    st.subheader("🎯 Our Mission")
    st.markdown("""
    <p style="text-align: justify;">
    <em style="font-size: 16.5px;"><b>Mano-Raaga</b> is an innovative music player that blends <b>emotion recognition</b> with <b>AI-powered recommendations</b>. My goal is to create an immersive and healing musical journey that understands how you feel — and responds with the perfect tune.

    Whether you're smiling, reflecting, or just feeling meh, Mano-Raaga captures your mood using facial expression analysis and instantly curates a playlist to match.</em>
    </p>
    """, unsafe_allow_html=True)

    st.subheader("👩‍💻 Who Am I?")
    st.markdown("""
    <em style="font-size: 16.5px;">I'm a passionate developer and a student with a deep love for both <b>music and technology</b>. This project was built as part of our academic pursuit and personal curiosity into how AI can improve mental well-being.</em>
    """, unsafe_allow_html=True)

    st.markdown("""
        <ul style="list-style-type:none;">
            <li style="margin-bottom: 10px;">👤 Developer: &nbsp;&nbsp;&nbsp;&nbsp;<b style="font-weight: 900;"> Darshan N Hulamani</b></li>
            <li style="margin-bottom: 10px;">🎓 Institution: &nbsp;&nbsp;&nbsp;&nbsp;<b style="font-weight: 900;"> JSS Shri Manjunatheshwara Institute of UG & PG Studies, Dharwad</b></li>
            <li style="margin-bottom: 10px;">📅 Project Year: &nbsp;<b style="font-weight: 900;"> 2025</b></li>
        </ul>
    """, unsafe_allow_html=True)

    st.subheader("🧰 Built With Love Using")
    st.markdown("""
        <p><ul style="list-style-type:none; font-weight: bold;">
            <li style="margin-bottom: 10px;">🐍 Python & Streamlit</li>
            <li style="margin-bottom: 10px;">🎵 OpenCV & FER (Facial Emotion Recognition)</li>
            <li style="margin-bottom: 10px;">🗃️ MySQL for song data</li>
            <li style="margin-bottom: 10px;">🎧 Custom curated MP3 libraries</li>
        </ul></p>
    """, unsafe_allow_html=True)

    st.subheader("🚀 My Vision")
    st.markdown("""
    <em style="font-size: 16.5px;">I believe that music can be a digital therapist, a mood enhancer, and a trusted companion. In the future, I hope to:</em>
    """, unsafe_allow_html=True)
    st.markdown("""
        <ul style="list-style-type:square; font-weight: bold;">
            <li style="margin-bottom: 10px;">Add voice emotion recognition</li>
            <li style="margin-bottom: 10px;">Introduce mood-based music learning</li>
            <li style="margin-bottom: 10px;">Expand to mobile platforms for on-the-go experiences</li>
        </ul>
    """, unsafe_allow_html=True)

    st.subheader("🎤 Got feedback or Vibes to Share?")
    st.markdown("""
    <em style="font-size: 16.5px;">I'd love to hear your thoughts — whether it's about the playlist, your mood, or how I can make Mano-Raaga even cooler.</em><br>
    Reach out at my <b>Email & Github Accounts</b>.
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(""" <a href="mailto:mano.raaga.by.darshan@gmail.com"><b style="font-weight: 900;">mano.raaga.by.darshan@gmail.com</b></a><br>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<a href="" target="_blank"><b style="font-weight: 900;">[your GitHub repo link]</b></a>""", unsafe_allow_html=True)

def music():
    if st.session_state['user']:
        # Create a container for the entire music player
        player_container = st.container()
        
        with player_container:
            st.title("Music Player")
            st.write("Welcome to the music section.")
            
            if st.session_state['emotion']:
                st.write(f"Your current emotion is {st.session_state['emotion']}.")
                st.title("🎵 Mano-Raaga 🎶")

                if not filtered_mp3_files:
                    st.warning(f"No songs found in {st.session_state.selected_playlist}.")
                    return

                # Current song info
                current_song_index = st.session_state.current_song_index
                current_song = filtered_mp3_files[current_song_index]
                song_path = os.path.join(song_folder, current_song)
                st.success(f"Now Playing: {current_song}")

                # Initialize playback if needed
                if not pygame.mixer.music.get_busy() and not st.session_state.paused:
                    play_song(song_path)

                # Get current playback position
                song_length = get_song_length(song_path)
                track_position = get_current_position()

                # Create a form for the seek bar
                new_position = st.slider(
                    "Seek",
                    0,
                    song_length,
                    track_position,
                    1,
                    key=f"seek_slider_{track_position}"
                )
                
                # Playback controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⏮️ Previous", key="prev_btn"):
                        if current_song_index > 0:
                            st.session_state.current_song_index -= 1
                            st.session_state.paused = False
                            pygame.mixer.music.stop()
                            st.rerun()
                with col2:
                    if st.session_state.is_playing:
                        if st.button("⏸ Pause", key="pause_btn"):
                            pause_song()
                            st.rerun()
                    else:
                        if st.button("▶️ Play", key="play_btn"):
                            resume_song()
                            st.rerun()
                with col3:
                    if st.button("⏭️ Next", key="next_btn"):
                        if current_song_index < len(filtered_mp3_files) - 1:
                            st.session_state.current_song_index += 1
                            st.session_state.paused = False
                            pygame.mixer.music.stop()
                            st.rerun()

                # Volume control
                volume = st.slider(
                    "Volume",
                    0.0,
                    1.0,
                    st.session_state.volume,
                    0.01,
                    key="volume_slider"
                )
                if volume != st.session_state.volume:
                    pygame.mixer.music.set_volume(volume)
                    st.session_state.volume = volume

                # Playlist display
                selected_song = st.radio(
                    "🎶 Up Next:",
                    filtered_mp3_files,
                    index=current_song_index,
                    key="playlist_selector"
                )
                
                # Handle song selection changes
                if selected_song != current_song:
                    st.session_state.current_song_index = filtered_mp3_files.index(selected_song)
                    pygame.mixer.music.stop()
                    st.rerun()

                # Auto-update the UI if music is playing
                # if st.session_state.is_playing and not st.session_state.paused:
                
                if st.session_state.current_page == "Music" and st.session_state.is_playing and not st.session_state.paused:
                    time.sleep(1)
                    st.rerun()

            else:
                st.write("Please complete emotion capture first.")
    else:
        st.warning("Please login first.")

# -------------------------------
# Logout Function
# -------------------------------
def logout():
    """Handle user logout and cleanup."""
    try:
        # Stop any playing music first
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            time.sleep(0.5)  # Give system time to release file handles

        # Clear session state
        st.session_state['user'] = None
        st.session_state['emotion'] = None
        st.session_state['capture'] = None
        st.session_state['is_playing'] = False
        st.session_state['paused'] = False
        st.session_state['track_position'] = 0
        st.session_state['start_time'] = 0
        st.session_state['selected_playlist'] = "None"
        
        # Cleanup temporary files
        initial_cleanup_temp_files()
        
    except Exception as e:
        st.error(f"Error during logout: {str(e)}")
    finally:
        st.rerun()

# -------------------------------
# Footer
# -------------------------------
def show_footer():
    st.markdown("---")
    st.markdown("<center>© 2025 Mano-Raaga | Developed by <span style='font-weight: bolder; color: cyan'><b>Darshan N Hulamani</b></span>. <br>All rights reserved!</center>", unsafe_allow_html=True)

# -------------------------------
# Main Logic
# -------------------------------
def main():
    initialize_session()

    if not st.session_state.get('emotion'):
        play_background_music()
        set_background_image("images/bg11_music.avif")
    else:
        stop_background_music()
        set_background_image("images/music_wave.webp")

    # Display logo and sidebar header
    display_logo()
    display_sidebar_header()

    # Page routing
    if st.session_state['user']:
        if st.session_state['emotion']:
            page = st.sidebar.radio("Navigation", ["Music", "Dashboard"])
            if st.sidebar.button("Logout"):
                logout()
            
            if page == "Music":
                st.session_state.current_page = "Music"
                music()
            elif page == "Dashboard":
                st.session_state.current_page = "None"
                dashboard()
        else:
            page = st.sidebar.radio("Navigation", ["Capture Emotion", "Music", "Dashboard"])
            if st.sidebar.button("Logout"):
                logout()
            
            if page == "Capture Emotion":
                capture()
            elif page == "Music":
                music()
            elif page == "Dashboard":
                dashboard()
    else:
        page = st.sidebar.radio("Navigation", ["Login", "Register", "Forgot Password", "Dashboard"])
        
        cleanup_temp_files()
        
        if page == "Login":
            login()
        elif page == "Register":
            register()
        elif page == "Forgot Password":
            forgot_password()
        elif page == "Dashboard":
            dashboard()
    
    show_footer()

if __name__ == "__main__":
    main()