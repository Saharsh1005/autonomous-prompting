import cohere
from pinecone import Pinecone, ServerlessSpec
import hashlib

def initialize_cohere(api_key):
    """
    Initialize the Cohere client.

    Args:
    api_key (str): Your Cohere API key.

    """
    return cohere.Client(api_key)

def initialize_pinecone(api_key, environment, index_name, dimension):
    """
    Initialize the Pinecone vector database and create an index if it doesn't exist.

    Args:
    api_key (str): Your Pinecone API key.
    environment (str): Pinecone environment region.
    index_name (str): Name of the Pinecone index.
    dimension (int): Dimension of the embeddings.

    Returns:
    pinecone.Index: The initialized Pinecone index.
    """
    pc = Pinecone(api_key=api_key)
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=environment,
                region='us-east-1'
            )
        )
    return pc.Index(index_name)

# Generate embedding for text using Cohere
def generate_embedding(cohere_client, text, model="embed-english-v3.0"):
    """
    Generate an embedding for the given text using Cohere.

    Args:
    cohere_client (cohere.Client): The Cohere client.
    text (str): The input text to embed.
    model (str): The embedding model to use.

    Returns:
    list: The embedding vector.
    """
    response = cohere_client.embed(texts=[text],input_type="search_query",model=model)
    return response.embeddings[0]

def generate_id_from_question(question):
    """
    Generate a unique ID for a question using SHA-256 hashing.

    Args:
    question (str): The input question.

    Returns:
    str: A unique hash-based ID.
    """
    return hashlib.sha256(question.encode('utf-8')).hexdigest()

# Store embeddings in the Pinecone vector database
def store_embeddings(index, dataset, cohere_client, batch_size=100):
    """
    Store embeddings and metadata in the Pinecone vector database.

    Args:
    index (pinecone.Index): The Pinecone index.
    dataset (list of dict): Dataset containing 'question', 'prompt', 'answer', and 'priority'.
    cohere_client (cohere.Client): The Cohere client.
    """
    grouped = dataset.groupby('question')
    vectors = []
    for question, group in grouped:
        #select the lowest priority correct answer
        correct_answers = group[group['generated_answer'] == group['correct_answer']]
        if correct_answers.empty:
            continue  # Skip if no correct answers

        # Select the row with the lowest priority correct answer
        correct_row = correct_answers.sort_values('priority').iloc[0]
        #metadata
        metadata = {
            "question": question,  # Store the question as part of the metadata
            "strategy": correct_row['strategy'],
            "priority": int(correct_row['priority'])
        }
        embedding = generate_embedding(cohere_client, question)
        id = generate_id_from_question(question)
        vectors.append((id, embedding, metadata))
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []  # Reset the batch

    if vectors:
        index.upsert(vectors)
        
def retrieve_top_k_questions(index, cohere_client, new_question, top_k=5):
    """
    Retrieve the top-K most similar questions for a given new question.

    Args:
    index (pinecone.Index): The Pinecone index.
    cohere_client (cohere.Client): The Cohere client.
    new_question (str): The input question.
    top_k (int): Number of top similar questions to retrieve.

    Returns:
    list of dict: Retrieved questions with metadata and relevance scores.
    """
    # Generate embedding for the new question
    query_embedding = generate_embedding(cohere_client, new_question)

    # Query the vector database
    results = index.query(vector=query_embedding,top_k=5,include_metadata=True)
    # Extract and format results
    retrieved_questions = []
    for match in results['matches']:
        retrieved_questions.append({
            "question": match['metadata'].get('question', 'Unknown Question'),
            "strategy": match['metadata']['strategy'],
            "priority": match['metadata']['priority'],
            "score": match['score']
        })

    # Sort grouped results by score
    return sorted(retrieved_questions, key=lambda x: x['score'], reverse=True)