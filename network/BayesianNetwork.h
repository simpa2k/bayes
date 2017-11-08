//
// Created by Simon Olofsson on 2017-11-05.
//

#ifndef BAYES_BAYESIANNETWORK_H
#define BAYES_BAYESIANNETWORK_H


#include <armadillo>
#include <memory>

#include "../classifier/Classifier.h"
#include "../directedGraph/Graph.h"

using namespace std;
using namespace arma;

class BayesianNetwork {

    Classifier classifier;
    Graph<string, mat> graph;
    unordered_map<string, shared_ptr<umat>> dataByNode;

public:

    explicit BayesianNetwork(shared_ptr<mt19937> engine);

    bool add(const string& nodeIdentifier);
    bool record(const string& nodeIdentifier, const umat& data);
    umat get(const string& nodeIdentifier);
    bool connect(const string& visibleIdentifier, const string& hiddenIdentifier);
    vector<string> getModel();
    shared_ptr<mat> imputeHiddenNode();

};


#endif //BAYES_BAYESIANNETWORK_H
