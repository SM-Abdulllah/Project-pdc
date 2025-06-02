#!/bin/bash

echo "=== Apriori Algorithm Performance Testing ==="

# Compile all versions
echo "Compiling programs..."
g++ -o sequential aprioriomp.cpp -std=c++11
g++ -o parallel recursiveparallel.cpp -fopenmp -std=c++11
mpic++ -o distributed distributed.cpp -std=c++11

# Create sample data if it doesn't exist
if [ ! -f "sample_data.txt" ]; then
    echo "Creating sample dataset..."
    cat > sample_data.txt << EOF
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
fi

MIN_SUPPORT=2
DATA_FILE="sample_data.txt"

echo "Dataset: $DATA_FILE"
echo "Minimum Support: $MIN_SUPPORT"
echo "================================"

# Test Sequential Version
echo "Testing Sequential Version..."
echo -e "$DATA_FILE\n$MIN_SUPPORT" | ./sequential > sequential_output.txt 2>&1

# Test Parallel Version with different thread counts
echo "Testing Parallel Version..."
for threads in 1 2 4 8; do
    if [ $threads -le $(nproc) ]; then
        echo "  Testing with $threads threads..."
        echo -e "$DATA_FILE\n$MIN_SUPPORT\n$threads\n1" | ./parallel > parallel_${threads}_output.txt 2>&1
    fi
done

# Test Distributed Version with different process counts
echo "Testing Distributed Version..."
for procs in 1 2 4; do
    echo "  Testing with $procs processes..."
    echo -e "$MIN_SUPPORT\n$DATA_FILE\n1" | mpirun -np $procs ./distributed > distributed_${procs}_output.txt 2>&1
done

echo "================================"
echo "Performance testing completed!"
echo "Check output files for results:"
echo "  - sequential_output.txt"
echo "  - parallel_*_output.txt"
echo "  - distributed_*_output.txt"
echo "  - *_results.txt (timing data)"
echo "  - *_performance.txt (performance logs)"
