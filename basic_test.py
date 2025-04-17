# Example 1: Basic Usage with Document Embeddings
import numpy as np
from simple_vectors import VectorDatabase

def basic_usage_example():
    """
    Basic example showing how to create, add to, and query a vector database
    with document embeddings.
    """
    print("\n=== BASIC USAGE EXAMPLE ===")
    
    # Initialize the database
    db = VectorDatabase("./data")
    
    # Create some document embeddings (in real life, these would come from an embedding model)
    # Each document is represented as a 5-dimensional vector for simplicity
    documents = [
        "How to build a vector database from scratch",
        "Understanding vector embeddings and similarity search",
        "Python implementation of cosine similarity",
        "Building efficient data structures for machine learning",
        "How to normalize vectors for better search results"
    ]
    
    # These would normally come from an embedding model
    dummy_embeddings = [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # doc 1
        np.array([0.2, 0.3, 0.4, 0.5, 0.6]),  # doc 2
        np.array([0.3, 0.1, 0.2, 0.8, 0.1]),  # doc 3
        np.array([0.5, 0.5, 0.2, 0.1, 0.8]),  # doc 4
        np.array([0.1, 0.3, 0.2, 0.4, 0.6])   # doc 5
    ]
    
    # Add document embeddings to the database with metadata
    for i, (text, embedding) in enumerate(zip(documents, dummy_embeddings)):
        db.add_vector(
            embedding, 
            {
                "text": text,
                "id": i,
                "length": len(text)
            },
            normalize=True
        )
    
    # Query for similar documents
    query_embedding = np.array([0.2, 0.2, 0.3, 0.4, 0.5])  # Similar to documents about vector databases
    results = db.top_cosine_similarity(query_embedding, top_n=3)
    
    # Display results
    print("\nQuery results for 'vector database' concept:")
    for doc_id, metadata, similarity in results:
        print(f"Document: '{metadata['text']}'")
        print(f"Similarity: {similarity:.4f}")
        print(f"ID: {doc_id}")
        print("-" * 50)
    
    # Save to disk
    db.save_to_disk("documents")
    print("\nDatabase saved to disk as 'documents.svdb'")

    # Load from disk
    new_db = VectorDatabase("./data")
    new_db.load_from_disk("documents")
    print(f"Database loaded with {len(new_db.vectors)} vectors")

# Example 2: Working with Product Embeddings
def product_embedding_example():
    """
    Example showing how to use the vector database for product recommendations.
    """
    print("\n=== PRODUCT EMBEDDING EXAMPLE ===")
    
    # Initialize the database
    db = VectorDatabase("./data")
    
    # Product catalog with embeddings (in real scenarios, these would come from a model)
    products = [
        {"id": "p1", "name": "Blue T-shirt", "category": "clothing", "color": "blue", "embedding": np.array([0.2, 0.1, 0.8, 0.1, 0.3])},
        {"id": "p2", "name": "Red Sweater", "category": "clothing", "color": "red", "embedding": np.array([0.3, 0.2, 0.7, 0.2, 0.4])},
        {"id": "p3", "name": "Blue Jeans", "category": "clothing", "color": "blue", "embedding": np.array([0.25, 0.15, 0.75, 0.2, 0.3])},
        {"id": "p4", "name": "Smartphone", "category": "electronics", "color": "black", "embedding": np.array([0.8, 0.7, 0.1, 0.9, 0.2])},
        {"id": "p5", "name": "Laptop", "category": "electronics", "color": "silver", "embedding": np.array([0.85, 0.75, 0.05, 0.8, 0.1])},
        {"id": "p6", "name": "Headphones", "category": "electronics", "color": "black", "embedding": np.array([0.7, 0.65, 0.15, 0.75, 0.3])},
        {"id": "p7", "name": "Coffee Table", "category": "furniture", "color": "brown", "embedding": np.array([0.4, 0.8, 0.3, 0.1, 0.7])},
        {"id": "p8", "name": "Bookshelf", "category": "furniture", "color": "brown", "embedding": np.array([0.35, 0.85, 0.25, 0.15, 0.75])}
    ]
    
    # Add products to the database
    for product in products:
        db.add_vector(
            product["embedding"],
            {
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "color": product["color"]
            },
            normalize=True
        )
    
    # Get similar products to a blue t-shirt
    blue_tshirt = products[0]["embedding"]
    results = db.top_cosine_similarity(blue_tshirt, top_n=3)
    
    print("\nProducts similar to 'Blue T-shirt':")
    for doc_id, metadata, similarity in results:
        print(f"Product: {metadata['name']}")
        print(f"Category: {metadata['category']}")
        print(f"Color: {metadata['color']}")
        print(f"Similarity: {similarity:.4f}")
        print("-" * 50)
    
    # Filter by category
    def category_filter(doc_id, metadata):
        return metadata["category"] == "electronics"
    
    # Find electronics similar to a laptop
    laptop = products[4]["embedding"]
    electronics_results = db.top_cosine_similarity(
        laptop, 
        top_n=2,
        filter_func=category_filter
    )
    
    print("\nElectronics similar to 'Laptop':")
    for doc_id, metadata, similarity in electronics_results:
        print(f"Product: {metadata['name']}")
        print(f"Category: {metadata['category']}")
        print(f"Similarity: {similarity:.4f}")
        print("-" * 50)
    
    # Query by metadata
    blue_products = db.query_by_metadata(color="blue")
    
    print("\nAll blue products:")
    for doc_id, vector, metadata in blue_products:
        print(f"Product: {metadata['name']}")
        print(f"Category: {metadata['category']}")
        print("-" * 50)

# Example 3: Update and Delete Operations
def update_delete_example():
    """
    Example demonstrating how to update and delete vectors in the database.
    """
    print("\n=== UPDATE AND DELETE EXAMPLE ===")
    
    # Initialize the database
    db = VectorDatabase("./data")
    
    # Add some vectors
    vectors_with_meta = [
        (np.array([0.1, 0.2, 0.3]), {"name": "Vector 1", "tag": "test"}),
        (np.array([0.4, 0.5, 0.6]), {"name": "Vector 2", "tag": "test"}),
        (np.array([0.7, 0.8, 0.9]), {"name": "Vector 3", "tag": "production"})
    ]
    
    # Add vectors and get their IDs
    ids = db.add_vectors_batch(vectors_with_meta, normalize=True)
    
    print(f"Added {len(ids)} vectors with IDs: {ids}")
    
    # Update a vector
    db.update_vector(
        ids[0],
        new_vector=np.array([0.9, 0.8, 0.7]),
        new_metadata={"name": "Updated Vector 1", "tag": "test", "updated": True}
    )
    
    # Delete a vector
    db.delete_vector(ids[1])
    
    # Check remaining vectors
    print("\nAfter update and delete:")
    for i, (vector, metadata, vec_id) in enumerate(zip(db.vectors, db.metadata, db.ids)):
        print(f"Vector {i+1}:")
        print(f"  ID: {vec_id}")
        print(f"  Metadata: {metadata}")
        print(f"  Vector: {vector[:3]}...")
        print("-" * 50)
    
    # Get statistics
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

# Example 4: Performance with larger dataset
def performance_example():
    """
    Example demonstrating performance with a larger dataset.
    """
    print("\n=== PERFORMANCE EXAMPLE ===")
    
    # Initialize the database
    db = VectorDatabase("./data")
    
    # Generate a larger dataset (1000 random 100-dimensional vectors)
    num_vectors = 1000
    dim = 100
    
    print(f"Generating {num_vectors} random {dim}-dimensional vectors...")
    
    # Create vectors with random data
    import time
    start_time = time.time()
    
    for i in range(num_vectors):
        # Generate random vector
        vector = np.random.random(dim)
        
        # Add to database with metadata
        db.add_vector(
            vector,
            {
                "id": i,
                "group": i % 10,  # Assign to one of 10 groups
                "value": round(np.mean(vector), 3)  # Average value as metadata
            },
            normalize=True
        )
    
    add_time = time.time() - start_time
    print(f"Time to add {num_vectors} vectors: {add_time:.4f} seconds")
    
    # Compress vectors to save memory
    compression_ratio = db.compress_vectors(bits=16)
    print(f"Compressed vectors with ratio: {compression_ratio:.2f}x")
    
    # Query performance
    query_vector = np.random.random(dim)
    
    start_time = time.time()
    results = db.top_cosine_similarity(query_vector, top_n=10)
    query_time = time.time() - start_time
    
    print(f"Time to query top 10 similar vectors: {query_time:.4f} seconds")
    print(f"Average similarity of top results: {np.mean([sim for _, _, sim in results]):.4f}")
    
    # Test metadata filtering
    def group_filter(doc_id, metadata):
        return metadata["group"] == 5  # Only return vectors from group 5
    
    start_time = time.time()
    filtered_results = db.top_cosine_similarity(query_vector, top_n=5, filter_func=group_filter)
    filter_query_time = time.time() - start_time
    
    print(f"Time to query with filtering: {filter_query_time:.4f} seconds")
    print(f"All results are from group 5: {all(meta['group'] == 5 for _, meta, _ in filtered_results)}")
    
    # Save and load test
    start_time = time.time()
    db.save_to_disk("large_dataset")
    save_time = time.time() - start_time
    
    print(f"Time to save {num_vectors} vectors: {save_time:.4f} seconds")
    
    # Clear and reload
    db = VectorDatabase("./data")
    start_time = time.time()
    db.load_from_disk("large_dataset")
    load_time = time.time() - start_time
    
    print(f"Time to load {num_vectors} vectors: {load_time:.4f} seconds")
    
    # Get and print stats
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

# Example 5: Using the vector database with real text embeddings
def text_embedding_example():
    """
    Example showing how to use the database with real text embeddings.
    This requires having an embedding model installed.
    """
    print("\n=== TEXT EMBEDDING EXAMPLE ===")
    print("Note: This example requires sentence-transformers to be installed.")
    print("If you don't have it, install with: pip install sentence-transformers")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize the database
        db = VectorDatabase("./data")
        
        # Sample documents
        documents = [
            "Vector databases store and retrieve vectors efficiently",
            "Embeddings capture semantic meaning of text",
            "Machine learning models work with vector representations",
            "Cosine similarity measures the angle between vectors",
            "Python is a popular programming language for data science",
            "Neural networks transform inputs into vector embeddings",
            "Large language models use attention mechanisms",
            "Semantic search finds results based on meaning not just keywords",
            "Efficient algorithms speed up nearest neighbor search",
            "Normalization is important for vector comparison"
        ]
        
        print(f"Encoding {len(documents)} documents...")
        
        # Get embeddings for all documents
        embeddings = model.encode(documents)
        
        # Add to database
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            db.add_vector(
                embedding,
                {
                    "id": i,
                    "text": doc,
                    "tokens": len(doc.split())
                }
            )
        
        # Query with a new document
        query = "How do vector databases work with embeddings?"
        query_embedding = model.encode(query)
        
        results = db.top_cosine_similarity(query_embedding, top_n=3)
        
        print(f"\nQuery: '{query}'")
        print("\nTop 3 most similar documents:")
        for doc_id, metadata, similarity in results:
            print(f"Document: '{metadata['text']}'")
            print(f"Similarity: {similarity:.4f}")
            print("-" * 50)
            
    except ImportError:
        print("This example requires sentence-transformers, which is not installed.")
        print("You can run it after installing with: pip install sentence-transformers")

if __name__ == "__main__":
    # Run all examples
    basic_usage_example()
    product_embedding_example()
    update_delete_example()
    performance_example()
    
    # Uncomment to run the text embedding example (requires additional library)
    # text_embedding_example()