#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <cstring>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

class DistributedApriori {
private:
    int min_support;
    vector<vector<string>> local_transactions;
    map<vector<string>, int> frequent_itemsets;
    int rank, size;
    
    // Serialize itemset for MPI communication
    string serializeItemset(const vector<string>& itemset) {
        string result;
        for (size_t i = 0; i < itemset.size(); i++) {
            result += itemset[i];
            if (i < itemset.size() - 1) result += ",";
        }
        return result;
    }
    
    // Deserialize itemset from string
    vector<string> deserializeItemset(const string& str) {
        vector<string> itemset;
        stringstream ss(str);
        string item;
        
        while (getline(ss, item, ',')) {
            // Trim whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                itemset.push_back(item);
            }
        }
        
        return itemset;
    }
    
public:
    DistributedApriori(int min_sup) : min_support(min_sup) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    
    // Load and distribute data among processes
    void loadAndDistributeData(const string& filename) {
        vector<vector<string>> all_transactions;
        
        // Master process loads all data
        if (rank == 0) {
            ifstream file(filename);
            if (!file.is_open()) {
                cerr << "Error: Cannot open file " << filename << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            string line;
            while (getline(file, line)) {
                if (line.empty()) continue;
                
                vector<string> transaction;
                stringstream ss(line);
                string item;
                
                while (getline(ss, item, ',')) {
                    // Trim whitespace
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    
                    if (!item.empty() && item != "-1") {
                        transaction.push_back(item);
                    }
                }
                
                if (!transaction.empty()) {
                    sort(transaction.begin(), transaction.end());
                    all_transactions.push_back(transaction);
                }
            }
            file.close();
            
            cout << "Master loaded " << all_transactions.size() << " transactions" << endl;
        }
        
        // Broadcast total number of transactions
        int total_transactions = all_transactions.size();
        MPI_Bcast(&total_transactions, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (total_transactions == 0) {
            if (rank == 0) {
                cerr << "Error: No transactions loaded" << endl;
            }
            MPI_Finalize();
            exit(1);
        }
        
        // Calculate distribution
        int transactions_per_process = total_transactions / size;
        int remainder = total_transactions % size;
        
        // Distribute transactions using a simpler approach
        int local_start = rank * transactions_per_process + min(rank, remainder);
        int local_count = transactions_per_process + (rank < remainder ? 1 : 0);
        
        if (rank == 0) {
            // Master process distributes data
            for (int dest = 1; dest < size; dest++) {
                int dest_start = dest * transactions_per_process + min(dest, remainder);
                int dest_count = transactions_per_process + (dest < remainder ? 1 : 0);
                
                // Send count first
                MPI_Send(&dest_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                
                // Send transactions
                for (int i = 0; i < dest_count; i++) {
                    string trans_str = serializeItemset(all_transactions[dest_start + i]);
                    int str_len = trans_str.length();
                    MPI_Send(&str_len, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
                    MPI_Send(trans_str.c_str(), str_len, MPI_CHAR, dest, 2, MPI_COMM_WORLD);
                }
            }
            
            // Keep local portion for master
            for (int i = 0; i < local_count; i++) {
                local_transactions.push_back(all_transactions[i]);
            }
        } else {
            // Receive data from master
            MPI_Recv(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < local_count; i++) {
                int str_len;
                MPI_Recv(&str_len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                char* buffer = new char[str_len + 1];
                MPI_Recv(buffer, str_len, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[str_len] = '\0';
                
                string trans_str(buffer);
                local_transactions.push_back(deserializeItemset(trans_str));
                
                delete[] buffer;
            }
        }
        
        cout << "Process " << rank << " received " << local_transactions.size() << " transactions" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Generate local 1-itemsets
    map<string, int> generateLocalC1() {
        map<string, int> local_counts;
        
        for (const auto& transaction : local_transactions) {
            for (const string& item : transaction) {
                local_counts[item]++;
            }
        }
        
        return local_counts;
    }
    
    // Aggregate global 1-itemsets
    map<vector<string>, int> aggregateC1(const map<string, int>& local_counts) {
        map<vector<string>, int> global_candidates;
        
        // Collect all unique items across all processes
        set<string> local_items;
        for (const auto& pair : local_counts) {
            local_items.insert(pair.first);
        }
        
        // Convert to vector for MPI operations
        vector<string> local_items_vec(local_items.begin(), local_items.end());
        int local_item_count = local_items_vec.size();
        
        // Gather item counts from all processes
        vector<int> all_item_counts(size);
        MPI_Allgather(&local_item_count, 1, MPI_INT, all_item_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate total items and displacements
        int total_items = 0;
        vector<int> displacements(size);
        for (int i = 0; i < size; i++) {
            displacements[i] = total_items;
            total_items += all_item_counts[i];
        }
        
        // Prepare serialized local items
        string local_items_str;
        for (size_t i = 0; i < local_items_vec.size(); i++) {
            local_items_str += local_items_vec[i];
            if (i < local_items_vec.size() - 1) local_items_str += ",";
        }
        
        // Calculate string lengths for variable-length allgatherv
        vector<int> str_lengths(size);
        int local_str_len = local_items_str.length();
        MPI_Allgather(&local_str_len, 1, MPI_INT, str_lengths.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate displacements for strings
        vector<int> str_displacements(size);
        int total_str_len = 0;
        for (int i = 0; i < size; i++) {
            str_displacements[i] = total_str_len;
            total_str_len += str_lengths[i];
        }
        
        // Gather all items as strings
        char* all_items_buffer = new char[total_str_len + 1];
        MPI_Allgatherv(local_items_str.c_str(), local_str_len, MPI_CHAR,
                       all_items_buffer, str_lengths.data(), str_displacements.data(),
                       MPI_CHAR, MPI_COMM_WORLD);
        all_items_buffer[total_str_len] = '\0';
        
        // Parse all items and get unique set
        set<string> all_unique_items;
        string current_str;
        for (int i = 0; i < size; i++) {
            if (str_lengths[i] > 0) {
                string proc_items(all_items_buffer + str_displacements[i], str_lengths[i]);
                stringstream ss(proc_items);
                string item;
                while (getline(ss, item, ',')) {
                    if (!item.empty()) {
                        all_unique_items.insert(item);
                    }
                }
            }
        }
        
        delete[] all_items_buffer;
        
        // For each unique item, calculate global support
        for (const string& item : all_unique_items) {
            int local_count = (local_counts.find(item) != local_counts.end()) ? local_counts.at(item) : 0;
            int global_count = 0;
            
            MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            if (global_count >= min_support) {
                vector<string> itemset = {item};
                global_candidates[itemset] = global_count;
            }
        }
        
        return global_candidates;
    }
    
    // Generate candidates from frequent itemsets
    map<vector<string>, int> generateCandidates(const map<vector<string>, int>& frequent_k) {
        map<vector<string>, int> candidates;
        vector<vector<string>> itemsets;
        
        for (const auto& pair : frequent_k) {
            itemsets.push_back(pair.first);
        }
        
        for (size_t i = 0; i < itemsets.size(); i++) {
            for (size_t j = i + 1; j < itemsets.size(); j++) {
                if (itemsets[i].size() == 0) continue;
                
                // Check if first k-1 items are the same
                bool can_join = true;
                for (size_t k = 0; k < itemsets[i].size() - 1; k++) {
                    if (itemsets[i][k] != itemsets[j][k]) {
                        can_join = false;
                        break;
                    }
                }
                
                if (can_join) {
                    vector<string> candidate = itemsets[i];
                    candidate.push_back(itemsets[j].back());
                    sort(candidate.begin(), candidate.end());
                    candidates[candidate] = 0;
                }
            }
        }
        
        return candidates;
    }
    
    // Check if itemset is subset of transaction
    bool isSubset(const vector<string>& itemset, const vector<string>& transaction) {
        return includes(transaction.begin(), transaction.end(),
                       itemset.begin(), itemset.end());
    }
    
    // Count local support for candidates
    map<vector<string>, int> countLocalSupport(const map<vector<string>, int>& candidates) {
        map<vector<string>, int> local_support;
        
        for (const auto& pair : candidates) {
            int count = 0;
            for (const auto& transaction : local_transactions) {
                if (isSubset(pair.first, transaction)) {
                    count++;
                }
            }
            local_support[pair.first] = count;
        }
        
        return local_support;
    }
    
    // Aggregate global support
    map<vector<string>, int> aggregateSupport(const map<vector<string>, int>& local_support) {
        map<vector<string>, int> global_support;
        
        for (const auto& pair : local_support) {
            int global_count = 0;
            MPI_Allreduce(&pair.second, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            global_support[pair.first] = global_count;
        }
        
        return global_support;
    }
    
    // Filter by minimum support
    map<vector<string>, int> filterBySupport(const map<vector<string>, int>& candidates) {
        map<vector<string>, int> frequent;
        
        for (const auto& pair : candidates) {
            if (pair.second >= min_support) {
                frequent[pair.first] = pair.second;
            }
        }
        
        return frequent;
    }
    
    void runDistributedApriori() {
        auto start = high_resolution_clock::now();
        
        if (rank == 0) {
            cout << "\n=== Running Distributed Apriori Algorithm ===" << endl;
            cout << "Number of processes: " << size << endl;
            cout << "Minimum support: " << min_support << endl << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Generate local 1-itemsets
        auto local_c1 = generateLocalC1();
        
        // Aggregate to get global frequent 1-itemsets
        auto frequent_k = aggregateC1(local_c1);
        
        if (rank == 0) {
            cout << "Frequent 1-itemsets: " << frequent_k.size() << endl;
        }
        
        // Store frequent 1-itemsets
        for (const auto& pair : frequent_k) {
            frequent_itemsets[pair.first] = pair.second;
        }
        
        int k = 1;
        while (!frequent_k.empty()) {
            // Generate candidates for next level
            auto candidates = generateCandidates(frequent_k);
            if (candidates.empty()) break;
            
            if (rank == 0) {
                cout << "Generated " << candidates.size() << " candidates for level " << (k+1) << endl;
            }
            
            // Count local support
            auto local_support = countLocalSupport(candidates);
            
            // Aggregate global support
            auto global_support = aggregateSupport(local_support);
            
            // Filter by minimum support
            frequent_k = filterBySupport(global_support);
            
            if (rank == 0) {
                cout << "Frequent " << (k+1) << "-itemsets: " << frequent_k.size() << endl;
            }
            
            // Store frequent itemsets
            for (const auto& pair : frequent_k) {
                frequent_itemsets[pair.first] = pair.second;
            }
            
            k++;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        if (rank == 0) {
            cout << "\nDistributed Apriori completed!" << endl;
            cout << "Total frequent itemsets: " << frequent_itemsets.size() << endl;
            cout << "Execution time: " << duration.count() << " ms" << endl;
            
            // Save timing results
            ofstream result("distributed_results.txt", ios::app);
            result << "Distributed_" << size << "_processes" << endl << duration.count() << endl;
            result.close();
        }
    }
    
    void printResults() {
        if (rank == 0) {
            cout << "\n=== FREQUENT ITEMSETS ===" << endl;
            
            // Group by size for better readability
            map<int, vector<pair<vector<string>, int>>> grouped_results;
            
            for (const auto& pair : frequent_itemsets) {
                grouped_results[pair.first.size()].push_back(pair);
            }
            
            for (const auto& group : grouped_results) {
                cout << "\n" << group.first << "-itemsets:" << endl;
                cout << "-------------" << endl;
                
                for (const auto& pair : group.second) {
                    cout << "{ ";
                    for (size_t i = 0; i < pair.first.size(); i++) {
                        cout << pair.first[i];
                        if (i < pair.first.size() - 1) cout << ", ";
                    }
                    cout << " } : " << pair.second << endl;
                }
            }
        }
    }
    
    // Performance testing with different process counts
    void performanceTest() {
        if (rank == 0) {
            cout << "\n=== DISTRIBUTED PERFORMANCE TEST ===" << endl;
            cout << "Process Count: " << size << endl;
            cout << "Local Transactions per Process: " << local_transactions.size() << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        auto start = high_resolution_clock::now();
        
        // Run the algorithm
        auto local_c1 = generateLocalC1();
        auto frequent_k = aggregateC1(local_c1);
        
        for (const auto& pair : frequent_k) {
            frequent_itemsets[pair.first] = pair.second;
        }
        
        int k = 1;
        while (!frequent_k.empty()) {
            auto candidates = generateCandidates(frequent_k);
            if (candidates.empty()) break;
            
            auto local_support = countLocalSupport(candidates);
            auto global_support = aggregateSupport(local_support);
            frequent_k = filterBySupport(global_support);
            
            for (const auto& pair : frequent_k) {
                frequent_itemsets[pair.first] = pair.second;
            }
            
            k++;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        if (rank == 0) {
            cout << "Execution time with " << size << " processes: " << duration.count() << " ms" << endl;
            cout << "Total frequent itemsets found: " << frequent_itemsets.size() << endl;
            
            // Save performance results
            ofstream perf_log("distributed_performance.txt", ios::app);
            perf_log << "Processes: " << size << ", Time: " << duration.count() << " ms, Itemsets: " << frequent_itemsets.size() << endl;
            perf_log.close();
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int min_support;
    string filename;
    int mode;
    
    if (rank == 0) {
        cout << "=== Distributed Apriori Algorithm (MPI) ===" << endl;
        cout << "Enter minimum support count: ";
        cin >> min_support;
        
        cout << "Enter data filename: ";
        cin >> filename;
        
        cout << "Select mode:" << endl;
        cout << "1. Normal run" << endl;
        cout << "2. Performance test" << endl;
        cin >> mode;
    }
    
    // Broadcast parameters to all processes
    MPI_Bcast(&min_support, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast filename
    char filename_buffer[256];
    if (rank == 0) {
        strncpy(filename_buffer, filename.c_str(), 255);
        filename_buffer[255] = '\0';
    }
    MPI_Bcast(filename_buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    filename = string(filename_buffer);
    
    if (min_support <= 0) {
        if (rank == 0) {
            cerr << "Error: Minimum support must be positive" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    DistributedApriori apriori(min_support);
    apriori.loadAndDistributeData(filename);
    
    if (mode == 1) {
        apriori.runDistributedApriori();
        apriori.printResults();
    } else {
        apriori.performanceTest();
    }
    
    MPI_Finalize();
    return 0;
}
