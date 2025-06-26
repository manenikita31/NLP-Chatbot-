import spacy
import random
import pyttsx3
import speech_recognition as sr
import tkinter as tk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize TTS engine
engine = pyttsx3.init()

# Speech recognizer
recognizer = sr.Recognizer()

# Sample training data
training_sentences = [
    "hello", "hi", "hey", "good morning", "good evening",
    "bye", "goodbye", "see you later",
    "thanks", "thank you", "thx",
    "can you help me", "i need assistance", "i want support",
    "what is your name", "who are you",
    "how old are you", "your age"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting", "greeting",
    "goodbye", "goodbye", "goodbye",
    "thanks", "thanks", "thanks",
    "help", "help", "help",
    "name", "name",
    "age", "age"
]

responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
    "goodbye": ["Goodbye! Have a great day!", "See you later!"],
    "thanks": ["You're welcome!", "No problem!"],
    "help": ["Sure! How can I assist you?", "I'm here to help!"],
    "name": ["I'm a smart chatbot created with Python, spaCy, and scikit-learn."],
    "age": ["I'm timeless. I live in your computer!"],
    "default": ["Sorry, I didn't understand that. Could you rephrase?"]
}

# Preprocessing with spaCy
def preprocess(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmas)

# Train classifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([preprocess(sentence) for sentence in training_sentences])
clf = MultinomialNB()
clf.fit(X, training_labels)

# Predict intent
def predict_intent(user_input):
    processed = preprocess(user_input)
    X_test = vectorizer.transform([processed])
    predicted = clf.predict(X_test)[0]
    return predicted

# Generate bot reply
def get_response(intent):
    return random.choice(responses.get(intent, responses["default"]))

# Text-to-speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Process user input
def process_input(user_input):
    intent = predict_intent(user_input)
    reply = get_response(intent)
    return reply

# GUI actions
def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    reply = process_input(user_input)
    chat_log.insert(tk.END, "Bot: " + reply + "\n\n")
    speak(reply)
    entry.delete(0, tk.END)

def listen_microphone():
    try:
        with sr.Microphone() as source:
            chat_log.insert(tk.END, "Listening...\n")
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)
            chat_log.insert(tk.END, "You (voice): " + user_input + "\n")
            reply = process_input(user_input)
            chat_log.insert(tk.END, "Bot: " + reply + "\n\n")
            speak(reply)
    except sr.UnknownValueError:
        chat_log.insert(tk.END, "Sorry, I couldn't understand. Try again.\n\n")
    except sr.RequestError:
        chat_log.insert(tk.END, "Speech recognition service error.\n\n")

# GUI setup
root = tk.Tk()
root.title("Smart NLP Chatbot with Voice")
root.geometry("550x600")

chat_log = tk.Text(root, bd=1, bg="white", font=("Arial", 12), wrap=tk.WORD)
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

entry = tk.Entry(entry_frame, width=40, font=("Arial", 12))
entry.pack(side=tk.LEFT, padx=10)

send_button = tk.Button(entry_frame, text="Send", command=send_message, width=10, font=("Arial", 12))
send_button.pack(side=tk.LEFT)

voice_button = tk.Button(root, text="🎤 Speak", command=listen_microphone, width=15, font=("Arial", 12))
voice_button.pack(pady=10)

root.mainloop()
