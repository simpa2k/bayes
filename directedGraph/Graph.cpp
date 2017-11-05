//
// Created by simon on 2017-04-03.
//

#include "Graph.h"

/*template <typename T, typename W>
bool Graph<T, W>::add(T data) {

    typename std::map<T, node<T, W>>::iterator existing = nodes.find(data);

    if (existing != nodes.end()) {
        return false;
    }

    node<T, W>* newNode = new node<T, W>;
    newNode->data = data;

    return true;

}

template <typename T, typename W>
bool Graph<T, W>::connect(T node1, T node2, W weight) {

    node<T, W>* existing1 = nodes.find(node1);
    node<T, W>* existing2 = nodes.find(node2);

    if (existing1 == nodes.end() || existing2 == nodes.end()) {
        return false;
    }

    edge<T, W>* connection = new edge<T, W>;
    connection->target = existing2;
    connection->weight = weight;

    existing1->edges.push_back(*connection);

    return true;

}*/

