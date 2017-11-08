//
// Created by simon on 2017-04-03.
//

#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <map>
#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <unordered_map>

using namespace std;

template <typename T, typename W>
struct edge;

template <typename T, typename W>
struct node;

template <typename T, typename W>
struct edge {

    node<T, W>* target;
    W weight;

};

template <typename T, typename W>
struct node {

    T data;
    int indegree = 0;

    vector<edge<T, W>> edges;

};

template <class T, class W>
class Graph {

    unordered_map<T, node<T, W>> nodes;
    bool areConnected(const node<T, W>& first, const node<T, W>& second);
    typename vector<edge<T, W>>::iterator getConnection(const node<T, W>& first, const node<T, W>& second);

public:

    bool add(T data);
    bool connect(T node1, T node2, W weight);
    W* getWeight(const T& node1, const T& node2);
    void getWeight(const T& node1, const T& node2, const W& target);
    map<T, W> getWeights(const T& nodeKey);
    vector<T> topologicalSort();

};

template <typename T, typename W>
bool Graph<T, W>::add(T data) {

    typename unordered_map<T, node<T, W>>::iterator existing = nodes.find(data);

    if (existing != nodes.end()) {
        return false;
    }

    auto newNode = new node<T, W>;
    newNode->data = data;

    nodes[data] = *newNode;

    return true;

}

template <typename T, typename W>
bool Graph<T, W>::connect(T node1, T node2, W weight) {

    typename unordered_map<T, node<T, W>>::iterator existing1 = nodes.find(node1);
    typename unordered_map<T, node<T, W>>::iterator existing2 = nodes.find(node2);

    if (existing1 == nodes.end() || existing2 == nodes.end()) {
        return false;
    }

    vector<edge<T, W>> &edges = existing1->second.edges;

    edge<T, W>* connection;

    for (auto iter = edges.begin(); iter != edges.end(); ++iter) {

        if (iter->target->data == existing2->second.data) {

            connection = &(*iter);
            connection->weight = weight;

            return true;

        }
    }

    connection = new edge<T, W>;

    connection->target = &existing2->second;
    connection->weight = weight;

    existing1->second.edges.push_back(*connection);
    existing2->second.indegree++; // ToDo: can this handle more than two levels?

    return true;
}

template <typename T, typename W>
vector<T> Graph<T, W>::topologicalSort() {

    vector<T> topologicalOrdering;
    map<T, int> indegrees;
    queue<node<T, W>> queue;

    for (auto const& it : nodes) {

        if (it.second.indegree == 0) {
            queue.push(it.second);
        } else {
            indegrees[it.second.data] = it.second.indegree;
        }
    }

    while (!queue.empty()) {

        node<T, W> n = queue.front();
        queue.pop();

        topologicalOrdering.push_back(n.data);

        for (auto const& it : n.edges) {

            int* indegree = &indegrees[it.target->data];

            if (--*indegree == 0) {
                queue.push(*it.target);
            }
        }
    }
    return topologicalOrdering;

}

template <typename T, typename W>
bool Graph<T, W>::areConnected(const node<T, W>& first, const node<T, W>& second) {

    vector<edge<T, W>> &edges = first.edges;
    auto it = find_if(edges.begin(), edges.end(), [&second](const edge<T, W>& edge) {
        return edge.target->data == second.data;
    });

    return it != edges.end();
}

template<typename T, typename W>
typename vector<edge<T, W>>::iterator Graph<T, W>::getConnection(const node<T, W>& first, const node<T, W>& second) {

    vector<edge<T, W>> &edges = first.edges;

    auto it = find_if(edges.begin(), edges.end(), [&second](const edge<T, W>& edge) {
        return edge.target->data == second.data;
    });

    return it;
}

template<typename T, typename W>
W* Graph<T, W>::getWeight(const T& node1, const T& node2) {

    W* weight = NULL;

    typename unordered_map<T, node<T, W>>::iterator existing1 = nodes.find(node1);
    typename unordered_map<T, node<T, W>>::iterator existing2 = nodes.find(node2);

    if (existing1 == nodes.end() || existing2 == nodes.end() || !areConnected(existing1->second, existing2->second)) {
        return NULL;
    }

    weight = new W();

    vector<edge<T, W>> &edges = existing1->second.edges;

    for (auto iter = edges.begin(); iter != edges.end(); ++iter) {

        edge<T, W> edge = *iter;
        if (edge.target->data == existing2->second.data) {
            *weight = edge.weight;
            break;
        }
    }

    return weight;
}

template<typename T, typename W>
void Graph<T, W>::getWeight(const T& node1, const T& node2, const W& target) {

    typename unordered_map<T, node<T, W>>::iterator existing1 = nodes.find(node1);
    typename unordered_map<T, node<T, W>>::iterator existing2 = nodes.find(node2);

    if (existing1 == nodes.end() || existing2 == nodes.end() || !areConnected(existing1->second, existing2->second)) {
        return;
    }

    typename vector<edge<T, W>>::iterator edgeIter = getConnection(existing1->second, existing2->second);

    target = edgeIter->weight;
}

template<typename T, typename W>
map<T, W> Graph<T, W>::getWeights(const T& nodeKey) {

    typename unordered_map<T, node<T, W>>::iterator existing = nodes.find(nodeKey);

    map<T, W> weights;

    for_each(existing->second.edges.begin(), existing->second.edges.end(), [&weights] (edge<T, W> currentEdge) {
        weights.insert(pair<T, W>(currentEdge.target->data, currentEdge.weight));
    });

    return weights;
};

#endif //GRAPH_GRAPH_H
