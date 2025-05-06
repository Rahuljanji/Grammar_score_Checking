📢 Grammar Scoring Engine 🎙️📊
A machine learning-powered tool to evaluate spoken grammar using acoustic features and real-time feedback.

🚀 Overview
The Grammar Scoring Engine analyzes spoken audio to predict grammar accuracy using advanced acoustic features and machine learning models. It provides instant feedback via an interactive web interface, making it ideal for exam preparation, language learning, and professional communication enhancement.

✨ Features & Functionality
🔍 Audio Analysis
Extracts 20+ acoustic features, including:

MFCCs (Mel-frequency cepstral coefficients) – essential for speech recognition.

Spectral Contrast – helps distinguish different phonetic components.

Zero Crossing Rate (ZCR) – detects speech characteristics.

🤖 Machine Learning
Random Forest Regressor trained on a large speech dataset to predict grammar accuracy.

Provides real-time grammar scores (0.0 - 1.0 scale) to indicate spoken correctness.

📊 Visual Analytics
Waveform display for speech visualization.

MFCC heatmap for feature analysis.

Download full CSV report for detailed insights.

🎛️ Real-Time Feedback
Instant grammar assessment for spoken language.

Evaluates speech fluency and correctness with data-driven insights.

📂 File Support
Works with WAV & MP3 audio formats for seamless integration.

🏗️ Tech Stack
Built using cutting-edge tools:

Core Technologies

🎼 Librosa – Audio feature extraction

🤖 scikit-learn – Machine learning pipeline

🖥️ Streamlit – Interactive web interface

📊 NumPy/Pandas – Data processing

Supporting Libraries

📈 Matplotlib – Visualizations

💾 Joblib – Model persistence

✍️ LanguageTool-Python – Grammar validation

📚 Dataset
The model is trained using Mozilla's Common Voice dataset from Kaggle:

Annotated with speaker metadata & text transcripts

Curated Subset:

25,403 voice recordings (cv-invalid.csv)

MP3 audio files

Speaker metadata (age, gender, accent)

🔗 Source: Common Voice on Kaggle

🔎 Use Cases
🎓 Educational Applications
Exam Prep: Perfect for IELTS, TOEFL, and other spoken language assessments.

Personalized Learning: Tutors can analyze speech patterns to enhance lesson planning.

Standardized Testing: Useful for institutions conducting unbiased grammar assessments.

💼 Professional & Real-Time Uses
Interview Training: Helps job applicants refine spoken grammar.

Public Speaking & Podcasting: Improves fluency and clarity.

Call Center Training: Ensures professional communication.

AI Assistants: Enhances speech interpretation.
