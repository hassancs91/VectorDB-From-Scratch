import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import shutil
import argparse
import json

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
    for _ in range(10):  # Use 10 queries for benchmark
        # Generate a random query vector
        query_vector = np.random.randn(vectors.shape[1]).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        
        # Time the query
        start_time = time.time()
        _ = db.top_cosine_similarity(query_vector, top_n=10)
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

def benchmark_qdrant(vectors, metadata, batch_size, collection_name="test_collection", qdrant_config=None):
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
    
    # Initialize Qdrant client based on config
    if qdrant_config is None or qdrant_config.get("use_memory", True):
        # In-memory mode
        client = QdrantClient(":memory:")
        print("Using in-memory Qdrant")
    else:
        # Connect to external Qdrant server
        host = qdrant_config.get("host", "localhost")
        port = qdrant_config.get("port", 6333)
        client = QdrantClient(host=host, port=port)
        print(f"Connected to Qdrant at {host}:{port}")
    
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
    
    # No separate save/load time for Qdrant
    save_time = 0
    load_time = 0
    
    # Measure query time (average over multiple queries)
    query_times = []
    for _ in range(10):  # Use 10 queries for benchmark
        # Generate a random query vector
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        
        # Time the query
        start_time = time.time()
        _ = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=10
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
    
    # Get unique dimensions and databases
    dims = sorted(results_df['dim'].unique())
    dbs = results_df['database'].unique()
    
    # Set up the plot
    for dim in dims:
        # Filter for this dimension
        dim_df = results_df[results_df['dim'] == dim]
        
        for db in dbs:
            # Get data for this database
            db_data = dim_df[dim_df['database'] == db]
            if not db_data.empty:
                plt.plot(db_data['size'], db_data[metric], 'o-' if db == 'Custom' else 's--', 
                        label=f'{db} (dim={dim})')
    
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

def detailed_comparison_report(results_df):
    """Generate a detailed text report of the benchmark results"""
    # Summary dataframe with mean values
    summary = results_df.groupby(['database', 'dim']).agg({
        'vectors_per_second_insert': 'mean',
        'queries_per_second': 'mean',
        'save_time': 'mean',
        'load_time': 'mean'
    }).reset_index()
    
    print("\n=== BENCHMARK SUMMARY ===")
    
    # Database performance by dimension
    for dim in sorted(results_df['dim'].unique()):
        dim_data = summary[summary['dim'] == dim]
        print(f"\nDimension: {dim}")
        
        for _, row in dim_data.iterrows():
            db = row['database']
            print(f"  {db} DB:")
            print(f"    Insert Speed: {row['vectors_per_second_insert']:.2f} vectors/second")
            print(f"    Query Speed: {row['queries_per_second']:.2f} queries/second")
            if db == 'Custom':
                print(f"    Save Time: {row['save_time']:.4f} seconds (average)")
                print(f"    Load Time: {row['load_time']:.4f} seconds (average)")
    
    # Compare databases if both are present
    dbs = results_df['database'].unique()
    if 'Custom' in dbs and 'Qdrant' in dbs:
        print("\n=== CUSTOM DB vs QDRANT COMPARISON ===")
        
        for dim in sorted(results_df['dim'].unique()):
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
                print(f"  Insert Performance: Custom is {insert_ratio:.2f}x {'faster' if insert_ratio > 1 else 'slower'} than Qdrant")
                print(f"  Query Performance: Custom is {query_ratio:.2f}x {'faster' if query_ratio > 1 else 'slower'} than Qdrant")
    
    # Size scaling analysis
    print("\n=== SCALING ANALYSIS ===")
    
    # Get unique sizes sorted
    sizes = sorted(results_df['size'].unique())
    if len(sizes) >= 2:
        for db in results_df['database'].unique():
            print(f"\n{db} DB scaling with dataset size:")
            
            for dim in sorted(results_df['dim'].unique()):
                db_dim_data = results_df[(results_df['database'] == db) & (results_df['dim'] == dim)]
                
                if len(db_dim_data) >= 2:
                    # Get smallest and largest size data
                    small_size = db_dim_data[db_dim_data['size'] == min(sizes)]
                    large_size = db_dim_data[db_dim_data['size'] == max(sizes)]
                    
                    if not small_size.empty and not large_size.empty:
                        small_insert = small_size['vectors_per_second_insert'].values[0]
                        large_insert = large_size['vectors_per_second_insert'].values[0]
                        insert_scale = small_insert / large_insert if large_insert > 0 else float('inf')
                        
                        small_query = small_size['queries_per_second'].values[0]
                        large_query = large_size['queries_per_second'].values[0]
                        query_scale = small_query / large_query if large_query > 0 else float('inf')
                        
                        print(f"  Dimension {dim}:")
                        print(f"    Insert scaling: {insert_scale:.2f}x from {min(sizes):,} to {max(sizes):,} records")
                        print(f"    Query scaling: {query_scale:.2f}x from {min(sizes):,} to {max(sizes):,} records")
    
    return summary

def run_benchmark(vector_dims, dataset_sizes, batch_size, qdrant_config=None):
    """Run benchmarks with specified parameters"""
    # Create results dataframe
    results = []
    
    # For tracking progress
    total_tests = len(vector_dims) * len(dataset_sizes) * 2  # x2 for Qdrant and custom
    completed = 0
    
    for dim in vector_dims:
        print(f"\nBenchmarking with {dim} dimensions:")
        
        for size in dataset_sizes:
            print(f"  Dataset size: {size:,}")
            
            # Generate test data
            vectors, metadata = generate_test_data(size, dim)
            
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
                qdrant_results = benchmark_qdrant(vectors, metadata, batch_size, qdrant_config=qdrant_config)
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
    
    # Generate detailed report
    detailed_comparison_report(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Custom Vector DB vs Qdrant')
    
    # Dataset size parameters
    parser.add_argument('--min-size', type=int, default=1000, 
                        help='Minimum dataset size to test')
    parser.add_argument('--max-size', type=int, default=100000, 
                        help='Maximum dataset size to test')
    parser.add_argument('--size-steps', type=int, default=3, 
                        help='Number of dataset sizes to test between min and max (logarithmic scale)')
    
    # Vector dimension parameters
    parser.add_argument('--dimensions', type=str, default='128,384,768',
                        help='Comma-separated list of vector dimensions to test')
    
    # Batch size
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for vector insertion')
    
    # Qdrant connection parameters
    parser.add_argument('--qdrant-memory', action='store_true', default=True,
                        help='Use in-memory Qdrant (default)')
    parser.add_argument('--qdrant-server', action='store_true',
                        help='Connect to a local Qdrant server')
    parser.add_argument('--qdrant-host', type=str, default='localhost',
                        help='Qdrant server host (default: localhost)')
    parser.add_argument('--qdrant-port', type=int, default=6333,
                        help='Qdrant server port (default: 6333)')
    
    # Preset configurations
    parser.add_argument('--preset', type=str, choices=['quick', 'medium', 'full'], 
                        help='Use a preset configuration (quick, medium, or full)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset == 'quick':
        vector_dims = [128]
        dataset_sizes = [1000, 10000]
        batch_size = 1000
    elif args.preset == 'medium':
        vector_dims = [128, 384]
        dataset_sizes = [1000, 10000, 50000]
        batch_size = 1000
    elif args.preset == 'full':
        vector_dims = [128, 384, 768]
        dataset_sizes = [1000, 10000, 50000, 100000]
        batch_size = 5000
    else:
        # Use custom configuration
        vector_dims = [int(dim) for dim in args.dimensions.split(',')]
        
        # Generate logarithmically spaced dataset sizes
        if args.size_steps <= 1:
            dataset_sizes = [args.min_size]
        else:
            dataset_sizes = np.logspace(
                np.log10(args.min_size),
                np.log10(args.max_size),
                num=args.size_steps
            ).astype(int).tolist()
            # Remove duplicates that might occur due to rounding
            dataset_sizes = sorted(list(set(dataset_sizes)))
        
        batch_size = args.batch_size
    
    # Configure Qdrant connection
    if args.qdrant_server:
        qdrant_config = {
            "use_memory": False,
            "host": args.qdrant_host,
            "port": args.qdrant_port
        }
    else:
        qdrant_config = {"use_memory": True}
    
    # Print benchmark configuration
    print("=== BENCHMARK CONFIGURATION ===")
    print(f"Vector dimensions: {vector_dims}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Batch size: {batch_size}")
    print(f"Qdrant config: {'In-memory' if qdrant_config['use_memory'] else 'Server at ' + qdrant_config['host'] + ':' + str(qdrant_config['port'])}")
    
    # Save configuration to file
    config = {
        "vector_dims": vector_dims,
        "dataset_sizes": dataset_sizes,
        "batch_size": batch_size,
        "qdrant_config": qdrant_config
    }
    with open('benchmark_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run the benchmark
    print("\n=== STARTING BENCHMARK ===")
    run_benchmark(vector_dims, dataset_sizes, batch_size, qdrant_config)
    
    print("\nBenchmarking complete!")