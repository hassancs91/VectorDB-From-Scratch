# Vector Database Benchmark

A benchmarking tool for comparing the performance of a custom vector database implementation against Qdrant, a popular vector database solution.

## Introduction

This project provides a comprehensive benchmarking framework for evaluating vector database performance. It compares a custom vector database implementation (included in this repository) with Qdrant across various metrics:

- Vector insertion speed
- Query performance
- Save/load operations
- Scaling with dataset size and vector dimensions

The benchmark generates detailed reports and visualizations to help understand performance characteristics under different workloads.

## Prerequisites

### Python Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- tqdm

### Qdrant Installation

You can use Qdrant either in-memory (no installation required) or via Docker:

#### Using Docker (Recommended for production-like benchmarks)

1. Install Docker from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

2. Pull and run the Qdrant Docker image:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

This will start a Qdrant server accessible at `localhost:6333`.

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd vector-db-benchmark
```

2. Install the required Python packages:

```bash
pip install numpy pandas matplotlib tqdm
```

3. Install Qdrant client (optional, only needed if you want to benchmark against Qdrant):

```bash
pip install qdrant-client
```

## Usage

The benchmark can be run with various configurations to test different aspects of vector database performance.

### Basic Usage

Run the benchmark with default settings:

```bash
python benchmark.py
```

### Configuration Options

The benchmark supports several command-line arguments:

- `--min-size`: Minimum dataset size to test (default: 1000)
- `--max-size`: Maximum dataset size to test (default: 100000)
- `--size-steps`: Number of dataset sizes to test between min and max (default: 3)
- `--dimensions`: Comma-separated list of vector dimensions to test (default: "128,384,768")
- `--batch-size`: Batch size for vector insertion (default: 1000)
- `--preset`: Use a preset configuration ("quick", "medium", or "full")
- `--qdrant-memory`: Use in-memory Qdrant (default)
- `--qdrant-server`: Connect to a local Qdrant server
- `--qdrant-host`: Qdrant server host (default: "localhost")
- `--qdrant-port`: Qdrant server port (default: 6333)

### Example Commands

#### Quick Test (Small Datasets)

For a quick benchmark with small datasets:

```bash
python benchmark.py --preset quick
```

This will run tests with:
- Vector dimension: 128
- Dataset sizes: 1,000 and 10,000 vectors
- Batch size: 1,000

#### Medium Test

For a more comprehensive benchmark with moderate datasets:

```bash
python benchmark.py --preset medium
```

This will run tests with:
- Vector dimensions: 128 and 384
- Dataset sizes: 1,000, 10,000, and 50,000 vectors
- Batch size: 1,000

#### Full Benchmark

For a complete benchmark with larger datasets:

```bash
python benchmark.py --preset full
```

This will run tests with:
- Vector dimensions: 128, 384, and 768
- Dataset sizes: 1,000, 10,000, 50,000, and 100,000 vectors
- Batch size: 5,000

#### Custom Configuration

You can also specify a custom configuration:

```bash
python benchmark.py --dimensions 512,1024 --min-size 5000 --max-size 200000 --size-steps 4 --batch-size 2000
```

#### Using Docker-based Qdrant

To benchmark against a Qdrant server running in Docker:

1. Start the Qdrant Docker container as described in the Prerequisites section.

2. Run the benchmark with the `--qdrant-server` flag:

```bash
python benchmark.py --qdrant-server --preset medium
```

This will connect to the Qdrant server running at localhost:6333 instead of using the in-memory version.
