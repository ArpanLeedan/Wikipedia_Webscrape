from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import faiss
from urllib.parse import urlparse
import sys
sys.path.append('/home/arpanleedan/.local/lib/python3.10/site-packages')

app = Flask(__name__)

# Step 1: Web Scraping
def scrape_wikipedia_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve content. Status code:", response.status_code)
        return ""

    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        print("Main content div not found")
        return ""

    # Extract text from paragraphs within the main content div
    content = []
    for element in content_div.find_all('p'):
        text = element.get_text().strip()
        if text:
            content.append(text)

    if not content:
        print("No relevant content found")
        return ""

    return "\n\n".join(content)

# Step 2: Chunking
def chunk_content(content):
    chunks = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
    return chunks

# Step 3: Storing Chunks in Faiss Vector Database
def store_chunks_in_faiss(chunks, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(chunks)
    dim = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    vectors = tfidf_matrix.toarray().astype('float32')
    index.add(vectors)
    return index, chunks

# Step 4: LLM API
def call_llm_api(question, context):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Determine the length of the input text
    input_length = len(context.split())

    # Set max_length based on the input_length
    max_length = min(2 * input_length, 100)  # Adjust this value as needed

    # Call the summarizer with the updated max_length
    summary = summarizer(context, max_length=max_length, min_length=25, do_sample=False)[0]['summary_text']
    return summary

# Step 5: Question Retrieval and Chunk Matching
def retrieve_top_chunks(user_question, all_chunks, vectorizer):
    user_question_vector = vectorizer.transform([user_question]).toarray().astype('float32')
    chunk_vectors = vectorizer.transform(all_chunks).toarray().astype('float32')
    similarity_scores = cosine_similarity(user_question_vector, chunk_vectors).flatten()
    sorted_chunks_indices = similarity_scores.argsort()[::-1]
    top_chunks_indices = sorted_chunks_indices[:3]
    top_chunks = [all_chunks[i] for i in top_chunks_indices]
    return top_chunks


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.form
    user_question = data['question']
    url = data['wiki_url']

    # Parse the URL to extract the last part
    parsed_url = urlparse(url)
    last_part_of_url = parsed_url.path.split('/')[-1]

    # Extract the name mentioned in the question
    question_name = user_question.split('Who is ')[1].split('?')[0]

    # Check if the last part of the URL matches the name mentioned in the question
    if last_part_of_url.lower() != question_name.replace(' ', '_').lower():
        return jsonify({'answer': 'The name in the question does not match the URL.'})

    content = scrape_wikipedia_page(url)
    chunks = chunk_content(content)
    chunks = [chunk for chunk in chunks if chunk and not chunk.isspace()]

    if not chunks:
        return jsonify({'answer': 'No relevant information found in the provided URL.'})

    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    index, chunk_texts = store_chunks_in_faiss(chunks, vectorizer)
    top_chunks = retrieve_top_chunks(user_question, chunk_texts, vectorizer)
    context = " ".join(top_chunks)
    llm_answer = call_llm_api(user_question, context)

    return jsonify({'answer': llm_answer})


if __name__ == '__main__':
    app.run(debug=True, port=8080)
