import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import shutil
import argparse

# Import your vector database
from simple_vectors import VectorDatabase, SerializationFormat

# Qdrant client import (will be skipped if not installed)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Qdrant client not found. Install with: pip install qdrant-client")

# Constants
VECTOR_DIMS = [128, 384, 768]  # Different vector dimensions to test
DATASET_SIZES = [1000, 10000, 50000, 100000]  # Different dataset sizes
BATCH_SIZES = [100, 1000, 5000]  # Different batch sizes for insertion
QUERY_COUNTS = [10, 100]  # Number of queries to run for averaging
TOP_K = 10  # Number of results to retrieve in queries

def generate_test_data(size, dim):
    """Generate random vectors and metadata for testing"""
    # Generate random vectors
    vectors = np.random.randn(size, dim).astype(np.float32)
    
    # Normalize the vectors (both Qdrant and our DB work better with normalized vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors = vectors / norms
    
    # Generate simple metadata
    metadata = []
    for i in range(size):
        metadata.append({
            "id": str(i),
            "group": i % 10,
            "value": round(float(np.random.random()), 3)
        })
    
    return vectors, metadata

def benchmark_custom_db(vectors, metadata, batch_size, collection_name="test_collection"):
    """Benchmark operations on our custom vector database"""
    # Prepare the database path
    db_path = "./custom_db_benchmark"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    
    # Initialize the database
    db = VectorDatabase(db_path)
    
    # Measure insertion time
    total_vectors = len(vectors)
    start_time = time.time()
    
    # Insert in batches
    for i in range(0, total_vectors, batch_size):
        end_idx = min(i + batch_size, total_vectors)
        batch_vectors = vectors[i:end_idx]
        batch_metadata = metadata[i:end_idx]
        
        # Create batch in the format expected by add_vectors_batch
        batch = [(v, m) for v, m in zip(batch_vectors, batch_metadata)]
        db.add_vectors_batch(batch, normalize=False)  # Vectors are already normalized
    
    insertion_time = time.time() - start_time
    
    # Measure save time
    start_time = time.time()
    db.save_to_disk(collection_name)
    save_time = time.time() - start_time
    
    # Measure load time
    db = VectorDatabase(db_path)
    start_time = time.time()
    db.load_from_disk(collection_name)
    load_time = time.time() - start_time
    
    # Measure query time (average over multiple queries)
    query_times = []
    for _ in range(QUERY_COUNTS[0]):  # Use the smaller query count for initial testing
        # Generate a random query vector
        query_vector = np.random.randn(vectors.shape[1]).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        
        # Time the query
        start_time = time.time()
        _ = db.top_cosine_similarity(query_vector, top_n=TOP_K)
        query_times.append(time.time() - start_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    # Clean up
    shutil.rmtree(db_path)
    
    return {
        "insertion_time": insertion_time,
        "vectors_per_second_insert": total_vectors / insertion_time,
        "save_time": save_time,
        "load_time": load_time,
        "avg_query_time": avg_query_time,
        "queries_per_second": 1 / avg_query_time
    }

def benchmark_qdrant(vectors, metadata, batch_size, collection_name="test_collection"):
    """Benchmark operations on Qdrant"""
    if not QDRANT_AVAILABLE:
        return {
            "insertion_time": 0,
            "vectors_per_second_insert": 0,
            "save_time": 0,
            "load_time": 0,
            "avg_query_time": 0,
            "queries_per_second": 0
        }
    
    # Initialize Qdrant client (in-memory mode for fair comparison)
    client = QdrantClient(":memory:")
    
    # Create collection
    vector_dim = vectors.shape[1]
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE
        )
    )
    
    # Measure insertion time
    total_vectors = len(vectors)
    start_time = time.time()
    
    # Insert in batches
    for i in range(0, total_vectors, batch_size):
        end_idx = min(i + batch_size, total_vectors)
        batch_vectors = vectors[i:end_idx]
        batch_metadata = metadata[i:end_idx]
        
        # Create batch in the format expected by Qdrant
        batch_points = [
            models.PointStruct(
                id=j + i,  # unique ID
                vector=batch_vectors[j].tolist(),
                payload=batch_metadata[j]
            )
            for j in range(len(batch_vectors))
        ]
        
        # Upsert batch
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
    
    insertion_time = time.time() - start_time
    
    # No separate save/load time for in-memory Qdrant
    save_time = 0
    load_time = 0
    
    # Measure query time (average over multiple queries)
    query_times = []
    for _ in range(QUERY_COUNTS[0]):  # Use the smaller query count for initial testing
        # Generate a random query vector
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        
        # Time the query
        start_time = time.time()
        _ = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=TOP_K
        )
        query_times.append(time.time() - start_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    return {
        "insertion_time": insertion_time,
        "vectors_per_second_insert": total_vectors / insertion_time,
        "save_time": save_time,
        "load_time": load_time,
        "avg_query_time": avg_query_time,
        "queries_per_second": 1 / avg_query_time
    }

def plot_results(results_df, metric, title, filename):
    """Plot comparison results for a specific metric"""
    plt.figure(figsize=(12, 8))
    
    # Get unique dimensions and sizes
    dims = results_df['dim'].unique()
    sizes = results_df['size'].unique()
    
    # Set up the plot
    for dim in dims:
        # Filter for this dimension
        dim_df = results_df[results_df['dim'] == dim]
        
        # Get custom DB and Qdrant data
        custom_data = dim_df[dim_df['database'] == 'Custom'][['size', metric]]
        qdrant_data = dim_df[dim_df['database'] == 'Qdrant'][['size', metric]]
        
        # Plot
        plt.plot(custom_data['size'], custom_data[metric], 'o-', 
                label=f'Custom DB (dim={dim})')
        plt.plot(qdrant_data['size'], qdrant_data[metric], 's--',
                label=f'Qdrant (dim={dim})')
    
    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Dataset Size (records)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename)
    plt.close()

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks and generate visualizations"""
    # Create results dataframe
    results = []
    
    # For tracking progress
    total_tests = len(VECTOR_DIMS) * len(DATASET_SIZES) * 2  # x2 for Qdrant and custom
    completed = 0
    
    for dim in VECTOR_DIMS:
        print(f"\nBenchmarking with {dim} dimensions:")
        
        for size in DATASET_SIZES:
            print(f"  Dataset size: {size:,}")
            
            # Generate test data
            vectors, metadata = generate_test_data(size, dim)
            
            # Determine batch size based on dataset size
            if size <= 1000:
                batch_size = 100
            elif size <= 10000:
                batch_size = 1000
            else:
                batch_size = 5000
            
            # Benchmark custom DB
            print("    Testing custom vector database...")
            custom_results = benchmark_custom_db(vectors, metadata, batch_size)
            results.append({
                "database": "Custom",
                "dim": dim,
                "size": size,
                "batch_size": batch_size,
                **custom_results
            })
            completed += 1
            print(f"    Progress: {completed}/{total_tests} tests completed")
            
            # Benchmark Qdrant
            if QDRANT_AVAILABLE:
                print("    Testing Qdrant...")
                qdrant_results = benchmark_qdrant(vectors, metadata, batch_size)
                results.append({
                    "database": "Qdrant",
                    "dim": dim,
                    "size": size,
                    "batch_size": batch_size,
                    **qdrant_results
                })
            else:
                print("    Skipping Qdrant (not installed)")
                results.append({
                    "database": "Qdrant",
                    "dim": dim,
                    "size": size,
                    "batch_size": batch_size,
                    "insertion_time": 0,
                    "vectors_per_second_insert": 0,
                    "save_time": 0,
                    "load_time": 0, 
                    "avg_query_time": 0,
                    "queries_per_second": 0
                })
            completed += 1
            print(f"    Progress: {completed}/{total_tests} tests completed")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv('vector_db_benchmark_results.csv', index=False)
    print(f"\nResults saved to vector_db_benchmark_results.csv")
    
    # Plot results
    os.makedirs('benchmark_plots', exist_ok=True)
    
    # Insertion speed
    plot_results(
        results_df, 
        'vectors_per_second_insert', 
        'Vector Insertion Speed Comparison',
        'benchmark_plots/insertion_speed.png'
    )
    
    # Query speed
    plot_results(
        results_df, 
        'queries_per_second', 
        'Query Speed Comparison',
        'benchmark_plots/query_speed.png'
    )
    
    # Save time (just for custom DB as Qdrant in-memory has no save)
    custom_df = results_df[results_df['database'] == 'Custom']
    plot_results(
        custom_df, 
        'save_time', 
        'Save Time for Custom Vector DB',
        'benchmark_plots/save_time.png'
    )
    
    # Load time
    plot_results(
        custom_df, 
        'load_time', 
        'Load Time for Custom Vector DB',
        'benchmark_plots/load_time.png'
    )
    
    print(f"Benchmark visualizations saved to benchmark_plots/ directory")

def detailed_comparison_report(results_df):
    """Generate a more detailed text report of the benchmark results"""
    # Summary dataframe with mean values
    summary = results_df.groupby(['database', 'dim']).agg({
        'vectors_per_second_insert': 'mean',
        'queries_per_second': 'mean',
        'save_time': 'mean',
        'load_time': 'mean'
    }).reset_index()
    
    # Custom vs Qdrant performance ratio
    if QDRANT_AVAILABLE:
        for dim in VECTOR_DIMS:
            custom_data = summary[(summary['database'] == 'Custom') & (summary['dim'] == dim)]
            qdrant_data = summary[(summary['database'] == 'Qdrant') & (summary['dim'] == dim)]
            
            if not custom_data.empty and not qdrant_data.empty:
                custom_insert = custom_data['vectors_per_second_insert'].values[0]
                qdrant_insert = qdrant_data['vectors_per_second_insert'].values[0]
                insert_ratio = custom_insert / qdrant_insert if qdrant_insert > 0 else float('inf')
                
                custom_query = custom_data['queries_per_second'].values[0]
                qdrant_query = qdrant_data['queries_per_second'].values[0]
                query_ratio = custom_query / qdrant_query if qdrant_query > 0 else float('inf')
                
                print(f"\nDimension: {dim}")
                print(f"  Insert Speed: Custom is {insert_ratio:.2f}x {'faster' if insert_ratio > 1 else 'slower'} than Qdrant")
                print(f"  Query Speed: Custom is {query_ratio:.2f}x {'faster' if query_ratio > 1 else 'slower'} than Qdrant")
    
    # Size vs performance correlation
    size_perf = results_df.groupby(['database', 'size']).agg({
        'vectors_per_second_insert': 'mean',
        'queries_per_second': 'mean'
    }).reset_index()
    
    # Calculate scaling factors (how performance changes with size)
    for db in ['Custom', 'Qdrant']:
        db_data = size_perf[size_perf['database'] == db]
        if len(db_data) >= 2:
            sizes = db_data['size'].values
            inserts = db_data['vectors_per_second_insert'].values
            queries = db_data['queries_per_second'].values
            
            # Compare smallest to largest
            insert_scale = inserts[0] / inserts[-1]
            query_scale = queries[0] / queries[-1]
            
            print(f"\n{db} DB Scaling (small to large datasets):")
            print(f"  Insert Speed: {insert_scale:.2f}x slower at {sizes[-1]:,} records vs {sizes[0]:,} records")
            print(f"  Query Speed: {query_scale:.2f}x slower at {sizes[-1]:,} records vs {sizes[0]:,} records")
    
    return summary

def run_query_scaling_test(dim=128, size=10000):
    """Run a test to measure how query performance scales with number of results"""
    print(f"\nTesting query scaling with k (dimension={dim}, size={size})...")
    
    # Generate test data
    vectors, metadata = generate_test_data(size, dim)
    
    # Initialize databases
    db_path = "./scaling_test"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    
    custom_db = VectorDatabase(db_path)
    
    # Add vectors to custom DB
    batch = [(v, m) for v, m in zip(vectors, metadata)]
    custom_db.add_vectors_batch(batch, normalize=False)
    
    # Initialize Qdrant if available
    if QDRANT_AVAILABLE:
        qdrant_client = QdrantClient(":memory:")
        qdrant_client.recreate_collection(
            collection_name="scaling_test",
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        )
        
        # Add vectors to Qdrant
        batch_points = [
            models.PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload=metadata[i]
            )
            for i in range(len(vectors))
        ]
        qdrant_client.upsert(
            collection_name="scaling_test",
            points=batch_points
        )
    
    # Test values of k (result count)
    k_values = [1, 5, 10, 20, 50, 100, 200, 500]
    results = []
    
    # Generate query vectors
    num_queries = 10
    query_vectors = []
    for _ in range(num_queries):
        query_vector = np.random.randn(dim).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vectors.append(query_vector)
    
    # Test custom DB
    for k in k_values:
        query_times = []
        for query_vector in query_vectors:
            start_time = time.time()
            _ = custom_db.top_cosine_similarity(query_vector, top_n=k)
            query_times.append(time.time() - start_time)
        
        avg_time = sum(query_times) / len(query_times)
        results.append({
            'database': 'Custom',
            'k': k,
            'query_time': avg_time,
            'queries_per_second': 1 / avg_time
        })
    
    # Test Qdrant
    if QDRANT_AVAILABLE:
        for k in k_values:
            query_times = []
            for query_vector in query_vectors:
                start_time = time.time()
                _ = qdrant_client.search(
                    collection_name="scaling_test",
                    query_vector=query_vector.tolist(),
                    limit=k
                )
                query_times.append(time.time() - start_time)
            
            avg_time = sum(query_times) / len(query_times)
            results.append({
                'database': 'Qdrant',
                'k': k,
                'query_time': avg_time,
                'queries_per_second': 1 / avg_time
            })
    
    # Clean up
    shutil.rmtree(db_path)
    
    # Convert to DataFrame and plot
    scaling_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    for db in scaling_df['database'].unique():
        db_data = scaling_df[scaling_df['database'] == db]
        plt.plot(db_data['k'], db_data['query_time'], 'o-', label=db)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Results (k)')
    plt.ylabel('Query Time (seconds)')
    plt.title('Query Time vs Number of Results')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('benchmark_plots', exist_ok=True)
    plt.savefig('benchmark_plots/query_scaling.png')
    plt.close()
    
    # Save raw data
    scaling_df.to_csv('query_scaling_results.csv', index=False)
    print(f"Query scaling results saved to query_scaling_results.csv")
    print(f"Query scaling plot saved to benchmark_plots/query_scaling.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Custom Vector DB vs Qdrant')
    parser.add_argument('--quick', action='store_true', help='Run a quick benchmark with smaller dataset sizes')
    parser.add_argument('--scaling', action='store_true', help='Run query scaling test')
    args = parser.parse_args()
    
    if args.quick:
        # Use smaller datasets for quick testing
        DATASET_SIZES = [1000, 10000]
        VECTOR_DIMS = [128]
    
    print(f"Running benchmark with:")
    print(f"  Vector dimensions: {VECTOR_DIMS}")
    print(f"  Dataset sizes: {DATASET_SIZES}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    
    if not QDRANT_AVAILABLE:
        print("\nWARNING: Qdrant client not installed. Only testing custom DB.")
        print("Install with: pip install qdrant-client")
    
    # Run the benchmarks
    run_comprehensive_benchmark()
    
    # Load results and generate a detailed report
    if os.path.exists('vector_db_benchmark_results.csv'):
        results_df = pd.read_csv('vector_db_benchmark_results.csv')
        summary = detailed_comparison_report(results_df)
    
    # Run scaling test if requested
    if args.scaling:
        run_query_scaling_test()
    
    print("\nBenchmarking complete!")