//
// Created by Simon Olofsson on 2017-11-05.
//

#include "BayesianNetwork.h"

BayesianNetwork::BayesianNetwork(shared_ptr<mt19937> engine) : classifier(engine) {}

bool BayesianNetwork::add(string nodeIdentifier) {
    return true;
}

bool BayesianNetwork::record(string nodeIdentifier, umat data) {
    return false;
}

