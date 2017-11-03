//
// Note that this code is partly a translation of a Python example presented in an excellent blog
// post by Jace Kohlmeier at:
// http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy
//
// Apart from the translation, this code expands on the example by providing functionality for using
// an arbitrary amount of visible and hidden states.
//
// All credit goes to Jace Kohlmeier for the original algorithm.
//

#include <armadillo>
#include <memory>

using namespace std;
using namespace arma;

const int HIDDEN_STATES = 3;
const int VISIBLE_STATES = 2;

random_device r;
mt19937 engine(r());

void setEngine(mt19937 &eng) {
    engine = eng;
}

/**
 * Utility function for expanding a matrix vertically, copying the values in the zeroth row
 * into all new rows. This means it effectively only works for row vectors.
 *
 * @param target The matrix to be expanded.
 * @param targetRows The final amount of rows desired.
 */
void expandVertically(mat& target, int targetRows) {

    target.resize(targetRows, target.n_cols);

    for (int i = 1; i < target.n_rows; ++i) {
        target.row(i) = target.row(0);
    }
}

/**
 * Utility function for expanding a matrix horizontally, copying the values in the zeroth column
 * into all new columns. This means it effectively only works for column vectors.
 *
 * @param target The matrix to be expanded.
 * @param targetColumns The final amount of columns desired.
 */
void expandHorizontally(umat& target, int targetColumns) {

    target.resize(target.n_rows, targetColumns);

    for (int i = 0; i < target.n_rows; ++i) {

        double value = (target)(i, 0);
        target.row(i).fill(value);

    }
}

/**
 * Utility function for simulating data for a hidden node with an arbitrary amount of states.
 *
 * @param distribution Expected to be a row vector where each column represents the probability of activation for the
 * state represented by the column's index.
 * @param samples The amount of data points to be generated.
 * @param engine
 * @return a pointer to a row vector of values.
 */
shared_ptr<umat> simulateHiddenData(mat& distribution, const int samples, mt19937& engine) {

    discrete_distribution<> dist(distribution.begin(), distribution.end());
    rowvec hiddenData(samples);

    hiddenData.imbue([&] () {
        return dist(engine);
    });

    return make_shared<umat>(conv_to<umat>::from(hiddenData));
}

/**
 * Utility function for simulating data for a visible node, given the state of a hidden node.
 *
 * @param hiddenData A row vector where each column represents a data point.
 * @param distribution The distribution to be used. Expected to be a matrix where each row represents the state of
 * the visible node and each row the state of the hidden node.
 * @param engine
 * @return
 */
shared_ptr<umat> simulateVisibleData(const umat& hiddenData, const mat& distribution, mt19937& engine) {

    auto visibleData = make_shared<umat>(hiddenData);
    
    visibleData->transform([&] (int val) {

        colvec col = distribution.col(val);
        discrete_distribution<> dist(col.begin(), col.end());

        return dist(engine);

    });

    return make_shared<umat>(trans(*visibleData));
}

/**
 * Calculate the probability of activation for a hidden node with an arbitrary amount of possible states.
 *
 * @param hiddenData A row vector where each column represents a data point.
 * @return A row vector of probabilities where each column is the probability of activation for the state indicated
 * by its index.
 */
shared_ptr<mat> computeThetaHidden(const umat& hiddenData) {

    umat histogram = hist(conv_to<::rowvec>::from(hiddenData), HIDDEN_STATES);
    auto thetaHidden = make_shared<mat>(conv_to<mat>::from(histogram));

    *thetaHidden /= accu(*thetaHidden);

    return thetaHidden;
}

/**
 * Calculate the probability of activation for a single node given the state of a hidden node,
 * represented as a column vector where each row represents a data point.
 *
 * @param hiddenData A row vector where each column represents a data point.
 * @param visibleData A column vector where each row represents a data point.
 * @return
 */
shared_ptr<mat> computeThetaVisibleForNode(const umat& hiddenData, const umat& visibleData) {

    auto thetaVisible = make_shared<mat>(VISIBLE_STATES, HIDDEN_STATES, fill::zeros);

    for (int i = 0; i < hiddenData.n_cols; ++i) {
        ++((*thetaVisible)(visibleData(i), hiddenData(i)));
    }

    thetaVisible->each_col([] (colvec& col) {
        double total = accu(col);
        col /= total;
    });

    return thetaVisible;
}

/**
 * Calculate the probability of activation for a series of visible nodes, given the state of a hidden node.
 * Expects the visible data to be passed as a matrix where each column represents a node, each row a data point
 * and each value an activation of a certain state.
 *
 * @param hiddenData
 * @param visibleData
 * @return a vector of matrices of probabilities. Each row represents the state of the given node, each column
 * the state of the hidden node and each value the probability that the state represented by the given row is
 * active, providing the hidden node takes the state indicated by the column. I.e p(B(row)|A(column))
 */
shared_ptr<vector<mat>> computeThetaVisible(umat& hiddenData, umat& visibleData) {

    auto thetaVisible = make_shared<vector<mat>>();
    mat convertedVisibleData = conv_to<mat>::from(visibleData);
    
    convertedVisibleData.each_col([&] (colvec& col) {
        shared_ptr<mat> nodeTheta = computeThetaVisibleForNode(hiddenData, conv_to<umat>::from(col));
        thetaVisible->push_back(*nodeTheta);
    });

    return thetaVisible;
}

/**
 * Replace all values of a matrix with values from a vector matrices, where the replacement value is chosen from
 * a matrix indicated by the column of the value to be replaced. When the matrix is chosen, use the value to
 * be replaced as index into its row and the index of a hidden node as index into its column.
 *
 * Effectively replaces data points with their corresponding probabilities of activation, given the state of a
 * hidden node.
 *
 * @param dataVisible The data to replace.
 * @param thetaHidden The state of the hidden node.
 * @param thetaVisible A set of conditional probabilities for a set of visible nodes.
 * @return A mat where each value has been replaced with the value contained in the matrix at the
 * index indicated by the value's column, at the row in this matrix indicated by the value itself
 * and at the column in this matrix indicated by the state of the hidden node.
 */
shared_ptr<mat> replaceAllValues(umat& dataVisible, int thetaHidden, vector<mat>& thetaVisible) {

    auto replacedValues = make_shared<mat>(conv_to<mat>::from(dataVisible));

    for (int i = 0; i < replacedValues->n_cols; ++i) { // For each column.
        auto col = replacedValues->col(i); // Pick out the column.
        mat node = thetaVisible.at(i); // Pick out the correct node, indicated by the column.

        col.transform([&] (double val) {
            return node(val, thetaHidden); // Visible states are indicated by measurement value, hidden states by the state passed.
        });
    }

    return replacedValues;
}

/**
 * Function for guessing the probability of a hidden node's different states being active.
 *
 * @param dataVisible Measurements for visible nodes. Expected to be a matrix where columns represent different nodes
 * and rows represent data points.
 * @param thetaHidden The probabilities for the hidden node's different states being active.
 * @param thetaVisible The probabilities of the visible nodes taking certain states, given that the hidden node takes
 * certain states. Rows indicate visible states, columns indicate hidden states.
 * @param generateNewData Whether new data should be returned or not.
 * @return A matrix with an amount of columns equal to the amount of hidden states, where each rows represent one data
 * point. If no new data was generated, each value will represent a probability of activation for a hidden state. If
 * new data was generated the matrix is effectively a histogram of counts of whether or not that probability activated
 * a state.
 */
shared_ptr<mat> imputeHiddenNode(umat& dataVisible, mat& thetaHidden, vector<mat>& thetaVisible, bool generateNewData) {

    mat hidden; // Not initializing this to a shared_ptr since it would only have to be reset each time join_rows was called.

    /*
     * Apply Baye's theorem. Note that the denominator
     * is not calculated, since it's effectively constant
     * given that the hidden nodes are known.
     */
    for (int i = 0; i < thetaHidden.n_cols; ++i) {

        shared_ptr<mat> probVis = replaceAllValues(dataVisible, i, thetaVisible); // Replace each visible data point with the probability for it taking the value it has.
        mat probVisUnnorm = thetaHidden(i) * prod(*probVis, 1); // Calculate the numerator of Baye's theorem; p(Ai)p(B|Ai).

        hidden = join_rows(hidden, probVisUnnorm); // Add all the data points for the case where the hidden node had the current value to the solution.

    };

    if (generateNewData) {

        mat hiddenData(1, hidden.n_rows);
        for (int i = 0; i < hidden.n_rows; ++i) { // Each row represents a data point.

            rowvec row = hidden.row(i);
            row = row / accu(row); // Since Baye's theorem gives p(Ai) which does not take the exact probabilities of all other A into account, normalize the values of Ai-n so that they become mutually exclusive.

            discrete_distribution<> dist(row.begin(), row.end()); // Create a distribution.
            hiddenData(0, i) = dist(engine); // Generate a data point
        }
        hidden = hiddenData; // Replace the hidden probabilities with the generated data.

    }
    return make_shared<mat>(hidden);
}

/**
 * Function to to try and estimate a hidden node's activation given data on a set of visible nodes affected by it.
 * Uses expectation maximization to continuously try to improve guessing.
 *
 * @param dataHidden Data on a hidden node's activation. Expected to be a row vector with each column representing a
 * data point. Given a perfect model, the data can be random.
 * @param dataVisible Data on the activation of a set of visible nodes. Expected to be a matrix where each column
 * represents a node and each row represents a data point.
 * @param learningIterations The amount of times to apply the expectation maximization.
 * @return The estimated probabilities of the hidden node's states being active.
 */
shared_ptr<mat> learn(umat& dataHidden, umat& dataVisible, int learningIterations) {

    shared_ptr<mat> thetaHidden = computeThetaHidden(dataHidden); // Compute the probability of the hidden node taking all possible values.
    shared_ptr<vector<mat>> thetaVisible = computeThetaVisible(dataHidden, dataVisible); // Compute the probability of each visible node taking all possible values, given each value that the hidden node can take.

    /*
     * Use expectation maximization to improve the estimate of the hidden node's probability.
     */
    for (int i = 0; i < learningIterations; ++i) {

        dataHidden = conv_to<umat>::from(*imputeHiddenNode(dataVisible, *thetaHidden, *thetaVisible, true)); // Use Baye's theorem to get better data on the hidden node using the visible nodes.

        thetaHidden.reset(new mat(*computeThetaHidden(dataHidden))); // Compute the probability of the hidden node with the improved data.
        thetaVisible = computeThetaVisible(dataHidden, dataVisible); // Compute the probability of each visible node using the improved data.

    }
    return thetaHidden;
}
