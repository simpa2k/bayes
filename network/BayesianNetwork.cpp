//
// Created by Simon Olofsson on 2017-11-05.
//

#include "BayesianNetwork.h"

BayesianNetwork::BayesianNetwork(shared_ptr<mt19937> engine) : classifier(engine) {}

bool BayesianNetwork::add(const string& nodeIdentifier) {

    dataByNode[nodeIdentifier] = make_shared<umat>();
    return graph.add(nodeIdentifier);

}

bool BayesianNetwork::record(const string& nodeIdentifier, const umat& data) {

    shared_ptr<umat> existingData = dataByNode[nodeIdentifier];
    *existingData = join_rows(*existingData, data);

    return true; // ToDo: Make sure this returns false when appropriate.
}

umat BayesianNetwork::get(const string& nodeIdentifier) {
    return *dataByNode[nodeIdentifier];
}

bool BayesianNetwork::connect(const string &visibleIdentifier, const string &hiddenIdentifier) {

    if (!graph.connect(visibleIdentifier, hiddenIdentifier, mat())) {
        return false;
    }

    if (dataByNode.find(hiddenIdentifier)->second->n_elem != 0 && dataByNode.find(visibleIdentifier)->second->n_elem != 0) {

        shared_ptr<umat> hiddenData = dataByNode[hiddenIdentifier];
        shared_ptr<umat> visibleData = dataByNode[visibleIdentifier];

        shared_ptr<mat> thetaVisible = classifier.computeThetaVisibleForNode(*hiddenData, hiddenData->max() + 1, *visibleData);

        return graph.connect(visibleIdentifier, hiddenIdentifier, *thetaVisible);

    }

    return true;
}

vector<string> BayesianNetwork::getModel() {
    return graph.topologicalSort();
}

shared_ptr<mat> BayesianNetwork::imputeHiddenNode() {

    const int LEARNING_ITERATIONS = 1000;
    vector<string> model = getModel();

    umat hiddenData = *dataByNode[model.back()];
    model.pop_back();

    umat visibleData;

    for (std::vector<string>::const_iterator i = model.begin(); i != model.end(); ++i) {
        visibleData = join_rows(visibleData, *dataByNode[*i]); 
    }

    std::cout << hiddenData << std::endl;
    std::cout << visibleData << std::endl;
}
