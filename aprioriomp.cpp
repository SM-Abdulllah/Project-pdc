#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <set>

using namespace std;
using namespace std::chrono;

class SequentialApriori {
private:
    int min_support;
    vector<vector<string>> transactions;
    
public:
    SequentialApriori(int min_sup) : min_support(min_sup) {}
    
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
    
    // Generate frequent 1-itemsets
    map<vector<string>, int> generateFrequent1Itemsets() {
        map<string, int> item_counts;
        
        // Count individual items
        for (const auto& transaction : transactions) {
            for (const string& item : transaction) {
                item_counts[item]++;
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
        
        // Generate candidates by joining frequent itemsets
        for (size_t i = 0; i < itemsets.size(); i++) {
            for (size_t j = i + 1; j < itemsets.size(); j++) {
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
    
    // Count support for candidates
    map<vector<string>, int> countSupport(const map<vector<string>, int>& candidates) {
        map<vector<string>, int> support_counts;
        
        for (const auto& candidate_pair : candidates) {
            int count = 0;
            for (const auto& transaction : transactions) {
                if (isSubset(candidate_pair.first, transaction)) {
                    count++;
                }
            }
            support_counts[candidate_pair.first] = count;
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
    
    // Main Apriori algorithm
    map<vector<string>, int> runApriori() {
        auto start = high_resolution_clock::now();
        
        cout << "\n=== Running Sequential Apriori Algorithm ===" << endl;
        cout << "Total transactions: " << transactions.size() << endl;
        cout << "Minimum support: " << min_support << endl << endl;
        
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
            
            // Count support
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
        
        cout << "\nSequential Apriori completed!" << endl;
        cout << "Total frequent itemsets: " << all_frequent_itemsets.size() << endl;
        cout << "Execution time: " << duration.count() << " ms" << endl;
        
        // Save timing results
        ofstream result("sequential_results.txt", ios::app);
        result << "Sequential" << endl << duration.count() << endl;
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
};

int main() {
    string filename;
    int min_support;
    
    cout << "=== Sequential Apriori Algorithm ===" << endl;
    cout << "Enter data filename: ";
    cin >> filename;
    
    cout << "Enter minimum support count: ";
    cin >> min_support;
    
    if (min_support <= 0) {
        cerr << "Error: Minimum support must be positive" << endl;
        return 1;
    }
    
    SequentialApriori apriori(min_support);
    
    if (!apriori.loadTransactions(filename)) {
        return 1;
    }
    
    auto frequent_itemsets = apriori.runApriori();
    apriori.printResults(frequent_itemsets);
    
    return 0;
}
