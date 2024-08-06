from sentence_transformers import SentenceTransformer
import time

def chunk_text(text, chunk_size):
    """Chunks text into smaller pieces."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_sentences(sentences, model, chunk_size=256):
    """Processes sentences by chunking and encoding."""
    embeddings = []
    for sentence in sentences:
        
        chunks = chunk_text(sentence, chunk_size)
        chunk_embeddings = model.encode(chunks)

        average_embedding = chunk_embeddings.mean(axis=0)
        embeddings.append(average_embedding)
    return embeddings

start_time = time.time()

sentences = [
    
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = process_sentences(sentences, model)

for i, embedding in enumerate(embeddings):
    print(f"Embedding for transcript {i+1}: {embedding}")

tensor_size = len(embeddings)

print(tensor_size)
print("Process finished --- %s seconds ---" % (time.time() - start_time))
