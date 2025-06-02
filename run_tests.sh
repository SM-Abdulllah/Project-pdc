#!/bin/bash

echo "=== Enhanced Apriori Algorithm Performance Testing ==="

echo "Compiling programs..."
g++ -o sequential aprioriomp.cpp -std=c++11 -O2
g++ -o parallel recursiveparallel.cpp -fopenmp -std=c++11 -O2
mpic++ -o distributed distributed.cpp -std=c++11 -O2

echo "Creating test datasets..."

cat > small_data.txt << EOF
bread,milk,eggs
bread,butter
milk,eggs,cheese
bread,milk,butter
eggs,cheese
bread,milk,eggs,butter
milk,cheese
bread,eggs
butter,cheese
bread,milk
milk,eggs
bread,cheese
eggs,butter
milk,butter
bread,eggs,cheese
EOF

echo "Generating medium dataset (1000 transactions)..."
python3 -c "
import random
items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'yogurt', 'cereal', 'juice', 'coffee', 'tea', 'sugar', 'flour', 'rice', 'pasta', 'oil']
with open('medium_data.txt', 'w') as f:
    for _ in range(1000):
        # Random transaction size between 2-8 items
        trans_size = random.randint(2, 8)
        transaction = random.sample(items, trans_size)
        f.write(','.join(sorted(transaction)) + '\n')
"

# Large dataset (10000 transactions)
echo "Generating large dataset (10000 transactions)..."
python3 -c "
import random
items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'yogurt', 'cereal', 'juice', 'coffee', 'tea', 'sugar', 'flour', 'rice', 'pasta', 'oil', 'apple', 'banana', 'orange', 'tomato', 'onion', 'potato', 'chicken', 'beef', 'fish', 'salt']
with open('large_data.txt', 'w') as f:
    for _ in range(10000):
        # Random transaction size between 3-10 items
        trans_size = random.randint(3, 10)
        transaction = random.sample(items, trans_size)
        f.write(','.join(sorted(transaction)) + '\n')
"

# Performance testing function
run_performance_test() {
    local dataset=$1
    local min_support=$2
    local max_threads=$(nproc)
    
    echo "================================"
    echo "Testing with dataset: $dataset"
    echo "Minimum Support: $min_support"
    echo "Available CPU cores: $max_threads"
    echo "================================"
    
    # Clean previous results
    rm -f *_results.txt *_performance.txt
    
    # Test Sequential Version
    echo "Testing Sequential Version..."
    echo -e "$dataset\n$min_support" | timeout 60s ./sequential > sequential_${dataset%.*}_output.txt 2>&1
    
    # Test Parallel Version with different thread counts
    echo "Testing Parallel Version..."
    for threads in 1 2 4 8 16; do
        if [ $threads -le $max_threads ]; then
            echo "  Testing with $threads threads..."
            echo -e "$dataset\n$min_support\n$threads\n1" | timeout 60s ./parallel > parallel_${threads}_${dataset%.*}_output.txt 2>&1
        fi
    done
    
    # Test Distributed Version with different process counts
    echo "Testing Distributed Version..."
    for procs in 1 2 4 8; do
        if [ $procs -le $max_threads ]; then
            echo "  Testing with $procs processes..."
            echo -e "$min_support\n$dataset\n1" | timeout 60s mpirun -np $procs ./distributed > distributed_${procs}_${dataset%.*}_output.txt 2>&1
        fi
    done
    
    echo "Completed testing for $dataset"
}

# Run tests on different datasets
echo "Starting comprehensive performance testing..."

# Test small dataset
run_performance_test "small_data.txt" 2
run_performance_test "medium_data.txt" 50
run_performance_test "large_data.txt" 200

echo "================================"
echo "Generating Performance Summary..."
echo "================================"

cat > performance_summary.py << 'EOF'
import os
import re

def extract_timing(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Look for execution time
            match = re.search(r'Execution time: (\d+) ms', content)
            if match:
                return int(match.group(1))
    except FileNotFoundError:
        pass
    return None

def analyze_results():
    datasets = ['small_data', 'medium_data', 'large_data']
    
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 50)
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 30)
        
        # Sequential
        seq_time = extract_timing(f'sequential_{dataset}_output.txt')
        if seq_time is not None:
            print(f"Sequential: {seq_time} ms")
        
        # Parallel
        print("Parallel (threads -> time):")
        for threads in [1, 2, 4, 8, 16]:
            par_time = extract_timing(f'parallel_{threads}_{dataset}_output.txt')
            if par_time is not None:
                speedup = seq_time / par_time if seq_time and par_time > 0 else 0
                print(f"  {threads} threads: {par_time} ms (speedup: {speedup:.2f}x)")
        
        # Distributed
        print("Distributed (processes -> time):")
        for procs in [1, 2, 4, 8]:
            dist_time = extract_timing(f'distributed_{procs}_{dataset}_output.txt')
            if dist_time is not None:
                speedup = seq_time / dist_time if seq_time and dist_time > 0 else 0
                print(f"  {procs} processes: {dist_time} ms (speedup: {speedup:.2f}x)")

if __name__ == "__main__":
    analyze_results()
EOF

python3 performance_summary.py

echo ""
echo "Performance testing completed!"
echo "Check the following files for detailed results:"
echo "  - *_output.txt files for execution logs"
echo "  - Run 'python3 performance_summary.py' for analysis"
echo ""
echo "To verify correctness, compare the frequent itemsets"
echo "across all implementations - they should be identical."
