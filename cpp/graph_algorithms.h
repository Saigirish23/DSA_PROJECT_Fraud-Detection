



#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct NodeFeatures {
    string node_id;
    int degree;
    double clustering;
    double pagerank;
    double betweenness;
};


void run(const string& edge_file, int num_iterations = 100);


unordered_map<string, int> compute_degree(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes);

unordered_map<string, double> compute_clustering(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes);

unordered_map<string, double> compute_pagerank(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes,
    int iterations = 100, double damping = 0.85, double tol = 1e-6);

unordered_map<string, double> compute_betweenness(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes);

#endif
