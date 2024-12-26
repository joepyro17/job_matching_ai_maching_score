import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    """Clean and preprocess text using spaCy"""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


def calculate_matching_score(resume, job_description):
    """Calculate a similarity score between resume and job description"""
    # Preprocess texts
    resume_cleaned = preprocess_text(resume)
    job_description_cleaned = preprocess_text(job_description)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_cleaned, job_description_cleaned])

    # Compute Cosine Similarity
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(score[0][0] * 100, 2)  # Return as percentage


# Usage
resume_text = "Experienced software developer with expertise in Python, Go, and AI solutions."
job_description = "Looking for a software developer skilled in Python, machine learning, and AI"


match_score = calculate_matching_score(resume_text, job_description)
print(f"Match Score: {match_score}\n")

