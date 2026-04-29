# Research Paper Recommendation System

A simple NLP-based Research Paper Recommendation System built using Python, Flask, Pandas, and Scikit-learn. The system recommends similar research papers based on content similarity using TF-IDF vectorization and cosine similarity.

## Features

- Content-based recommendation system
- TF-IDF feature extraction
- Cosine similarity matching
- Simple Flask web interface
- NLP-based document similarity analysis

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## How It Works

1. Research paper abstracts are converted into numerical vectors using TF-IDF.
2. Cosine similarity is calculated between all papers.
3. The system recommends the most similar papers based on content similarity.

## Installation

```bash
pip install flask pandas scikit-learn
```

## Run the Project

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## Project Structure

```text
research-paper-recommendation-system/
│
├── app.py
├── README.md
```

## Future Improvements

- Add larger research paper datasets
- Integrate real-time paper APIs
- Add user authentication
- Improve recommendation accuracy using deep learning models

## Author

Vaishnavi Shetty
