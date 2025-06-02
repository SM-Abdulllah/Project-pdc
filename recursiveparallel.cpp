#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

using namespace std;
using namespace std::chrono;

class ParallelApriori {
private:
    int min_support;
    vector<vector<string>> transactions;
    int num_threads;
    
public:
    ParallelApriori(int min_sup, int threads = 0) : min_support(min_sup) {
        if (threads > 0) {
            num_threads = threads;
            omp_set_num_threads(threads);
        } else {
            num_threads = omp_get_max_threads();
        }
    }
    
    // Read transactions from file
    bool loadTransactions(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Cannot open file " << filename << endl;
            return false;
        }
        
        string line;
        transactions.clear();
        
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
                transactions.push_back(transaction);
            }
        }
        
        file.close();
        cout << "Loaded " << transactions.size() << " transactions" << endl;
        return true;
    }
    
    // Parallel generation of frequent 1-itemsets
    map<vector<string>, int> generateFrequent1Itemsets() {
        map<string, int> item_counts;
        
        // Parallel counting with reduction
        #pragma omp parallel
        {
            map<string, int> local_counts;
            
            #pragma omp for
            for (int i = 0; i < transactions.size(); i++) {
                for (const string& item : transactions[i]) {
                    local_counts[item]++;
                }
            }
            
            // Merge local counts into global counts
            #pragma omp critical
            {
                for (const auto& pair : local_counts) {
                    item_counts[pair.first] += pair.second;
                }
            }
        }
        
        // Filter by minimum support
        map<vector<string>, int> frequent_1_itemsets;
        for (const auto& pair : item_counts) {
            if (pair.second >= min_support) {
                vector<string> itemset = {pair.first};
                frequent_1_itemsets[itemset] = pair.second;
            }
        }
        
        return frequent_1_itemsets;
    }
    
    // Generate candidate itemsets from frequent k-itemsets
    map<vector<string>, int> generateCandidates(const map<vector<string>, int>& frequent_k) {
        map<vector<string>, int> candidates;
        vector<vector<string>> itemsets;
        
        // Extract itemsets
        for (const auto& pair : frequent_k) {
            itemsets.push_back(pair.first);
        }
        
        // Parallel candidate generation
        #pragma omp parallel
        {
            map<vector<string>, int> local_candidates;
            
            #pragma omp for
            for (int i = 0; i < itemsets.size(); i++) {
                for (int j = i + 1; j < itemsets.size(); j++) {
                    vector<string> candidate = itemsets[i];
                    
                    // Check if first k-1 items are the same
                    bool can_join = true;
                    if (candidate.size() > 0) {
                        for (size_t k = 0; k < candidate.size() - 1; k++) {
                            if (itemsets[i][k] != itemsets[j][k]) {
                                can_join = false;
                                break;
                            }
                        }
                    }
                    
                    if (can_join) {
                        candidate.push_back(itemsets[j].back());
                        sort(candidate.begin(), candidate.end());
                        local_candidates[candidate] = 0;
                    }
                }
            }
            
            // Merge local candidates
            #pragma omp critical
            {
                for (const auto& pair : local_candidates) {
                    candidates[pair.first] = 0;
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
    
    // Parallel support counting
    map<vector<string>, int> countSupport(const map<vector<string>, int>& candidates) {
        map<vector<string>, int> support_counts;
        
        // Initialize support counts
        for (const auto& pair : candidates) {
            support_counts[pair.first] = 0;
        }
        
        // Convert to vector for better parallel access
        vector<vector<string>> candidate_list;
        for (const auto& pair : candidates) {
            candidate_list.push_back(pair.first);
        }
        
        // Parallel support counting
        vector<vector<int>> thread_counts(num_threads, vector<int>(candidate_list.size(), 0));
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            #pragma omp for
            for (int i = 0; i < transactions.size(); i++) {
                for (int j = 0; j < candidate_list.size(); j++) {
                    if (isSubset(candidate_list[j], transactions[i])) {
                        thread_counts[thread_id][j]++;
                    }
                }
            }
        }
        
        // Aggregate results
        for (int i = 0; i < candidate_list.size(); i++) {
            int total_count = 0;
            for (int t = 0; t < num_threads; t++) {
                total_count += thread_counts[t][i];
            }
            support_counts[candidate_list[i]] = total_count;
        }
        
        return support_counts;
    }
    
    // Filter candidates by minimum support
    map<vector<string>, int> filterBySupport(const map<vector<string>, int>& candidates) {
        map<vector<string>, int> frequent;
        
        for (const auto& pair : candidates) {
            if (pair.second >= min_support) {
                frequent[pair.first] = pair.second;
            }
        }
        
        return frequent;
    }
    
    // Main parallel Apriori algorithm
    map<vector<string>, int> runApriori() {
        auto start = high_resolution_clock::now();
        
        cout << "\n=== Running Parallel Apriori Algorithm ===" << endl;
        cout << "Total transactions: " << transactions.size() << endl;
        cout << "Minimum support: " << min_support << endl;
        cout << "Number of threads: " << num_threads << endl << endl;
        
        map<vector<string>, int> all_frequent_itemsets;
        
        // Generate frequent 1-itemsets
        auto frequent_k = generateFrequent1Itemsets();
        cout << "Frequent 1-itemsets: " << frequent_k.size() << endl;
        
        // Add to all frequent itemsets
        for (const auto& pair : frequent_k) {
            all_frequent_itemsets[pair.first] = pair.second;
        }
        
        int k = 1;
        while (!frequent_k.empty()) {
            // Generate candidates for next level
            auto candidates = generateCandidates(frequent_k);
            if (candidates.empty()) break;
            
            cout << "Generated " << candidates.size() << " candidates for level " << (k+1) << endl;
            
            // Count support in parallel
            auto support_counts = countSupport(candidates);
            
            // Filter by minimum support
            frequent_k = filterBySupport(support_counts);
            
            cout << "Frequent " << (k+1) << "-itemsets: " << frequent_k.size() << endl;
            
            // Add to all frequent itemsets
            for (const auto& pair : frequent_k) {
                all_frequent_itemsets[pair.first] = pair.second;
            }
            
            k++;
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        
        cout << "\nParallel Apriori completed!" << endl;
        cout << "Total frequent itemsets: " << all_frequent_itemsets.size() << endl;
        cout << "Execution time: " << duration.count() << " ms" << endl;
        
        // Save timing results
        ofstream result("parallel_results.txt", ios::app);
        result << "Parallel_" << num_threads << "_threads" << endl << duration.count() << endl;
        result.close();
        
        return all_frequent_itemsets;
    }
    
    void printResults(const map<vector<string>, int>& frequent_itemsets) {
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
    
    // Performance testing with different thread counts
    void performanceTest() {
        cout << "\n=== PARALLEL PERFORMANCE TEST ===" << endl;
        
        vector<int> thread_counts = {1, 2, 4, 8, 16};
        ofstream perf_log("parallel_performance.txt");
        
        for (int threads : thread_counts) {
            if (threads > omp_get_max_threads()) continue;
            
            cout << "\nTesting with " << threads << " threads..." << endl;
            omp_set_num_threads(threads);
            num_threads = threads;
            
            auto start = high_resolution_clock::now();
            auto results = runApriori();
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);
            
            cout << "Time with " << threads << " threads: " << duration.count() << " ms" << endl;
            perf_log << "Time Taken " << duration.count() << " No of threads " << threads << "." << endl;
        }
        
        perf_log.close();
    }
};

int main() {
    string filename;
    int min_support;
    int num_threads;
    int mode;
    
    cout << "=== Parallel Apriori Algorithm (OpenMP) ===" << endl;
    cout << "Enter data filename: ";
    cin >> filename;
    
    cout << "Enter minimum support count: ";
    cin >> min_support;
    
    cout << "Enter number of threads (0 for auto): ";
    cin >> num_threads;
    
    cout << "Select mode:" << endl;
    cout << "1. Normal run" << endl;
    cout << "2. Performance test" << endl;
    cin >> mode;
    
    if (min_support <= 0) {
        cerr << "Error: Minimum support must be positive" << endl;
        return 1;
    }
    
    ParallelApriori apriori(min_support, num_threads);
    
    if (!apriori.loadTransactions(filename)) {
        return 1;
    }
    
    if (mode == 1) {
        auto frequent_itemsets = apriori.runApriori();
        apriori.printResults(frequent_itemsets);
    } else {
        apriori.performanceTest();
    }
    
    return 0;
}
