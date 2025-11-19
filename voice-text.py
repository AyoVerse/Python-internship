# voice_to_blog.py
# Requires: pip install openai-whisper librosa soundfile noisereduce rake-nltk textstat scipy numpy

import os
import whisper
import librosa
import soundfile as sf
import noisereduce as nr
from rake_nltk import Rake
import textstat
import numpy as np
from scipy.io.wavfile import write

# -------------------
# 0️⃣ Generate Test Audio (WAV tone)
# -------------------
def create_test_audio(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        print("Creating test audio...")
        samplerate = 16000
        duration = 2  # 2 seconds
        frequency = 440  # A4 tone
        t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        write(file_path, samplerate, audio.astype(np.float32))
        print(f"Test audio created at '{file_path}'")
    else:
        print(f"Audio file already exists at '{file_path}'")

def transcribe_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    sf.write("temp.wav", y_denoised, sr)
    
    # Load Whisper model
    model = whisper.load_model("base")
    result = model.transcribe("temp.wav")
    return result["text"]

def generate_blog(transcript):
    intro = "Introduction:\n" + transcript[:200] + "...\n\n"
    body = "Main Content:\n" + transcript[200:800] + "...\n\n"
    conclusion = "Conclusion:\n" + transcript[-200:] + "...\n\n"
    return intro + body + conclusion

# -------------------
# 3️⃣ SEO Optimization
# -------------------
def extract_keywords(text, num=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:num]

def suggest_title_meta(text):
    title = text[:60].strip() + "..."
    meta_description = text[:160].strip() + "..."
    return title, meta_description

def score_readability(text):
    return textstat.flesch_reading_ease(text)

# -------------------
# 4️⃣ Main Flow
# -------------------
if __name__ == "__main__":
    AUDIO_FILE = "samples/audio/sample1.wav"
    
    # Create test audio
    create_test_audio(AUDIO_FILE)
    
    print("Transcribing audio...")
    transcript = transcribe_audio(AUDIO_FILE)
    print("\nTranscript:\n", transcript[:500], "...\n")  # show first 500 chars
    
    print("Generating blog...")
    blog_post = generate_blog(transcript)
    print("\nBlog Post:\n", blog_post[:500], "...\n")  # show preview
    
    print("Extracting SEO keywords...")
    keywords = extract_keywords(blog_post)
    title, meta = suggest_title_meta(blog_post)
    readability = score_readability(blog_post)
    
    print("\nSEO Keywords:", keywords)
    print("Title Suggestion:", title)
    print("Meta Description:", meta)
    print("Readability Score:", readability)
    
    # Save blog to file
    with open("blog_output.txt", "w", encoding="utf-8") as f:
        f.write(blog_post)
    
    print("\nBlog saved to 'blog_output.txt'")
