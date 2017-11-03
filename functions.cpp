//
// Created by simon on 2017-10-28.
//

#include <armadillo>
#include <memory>

const int HIDDEN_STATES = 3;
const int VISIBLE_STATES = 2;
const int SAMPLES = 10000;

using namespace std;
using namespace arma;

random_device r;
mt19937 engine(r());

void setEngine(mt19937 &eng) {
    engine = eng;
}

void expandVertically(arma::mat* target, int targetRows) {

    target->resize(targetRows, target->n_cols);

    for (int i = 1; i < target->n_rows; ++i) {
        target->row(i) = target->row(0);
    }

}

void expandHorizontally(arma::umat* target, int targetColumns) {

    target->resize(target->n_rows, targetColumns);

    for (int i = 0; i < target->n_rows; ++i) {

        double value = (*target)(i, 0);
        target->row(i).fill(value);

    }
}

arma::umat simulateHiddenData(arma::mat distribution, const int samples, std::mt19937* engine) {

    std::discrete_distribution<> dist(distribution.begin(), distribution.end());
    arma::rowvec hiddenData(samples);

    hiddenData.imbue([&] () {
        return dist(*engine);
    });

    return arma::conv_to<arma::umat>::from(hiddenData);
}

/*arma::umat simulateVisibleData(arma::umat dataHidden, arma::mat thetaVisible, const int samples) {

    arma::mat colZero = arma::trans(thetaVisible.col(0));
    arma::mat colOne = arma::trans(thetaVisible.col(1));

    expandHorizontally(&dataHidden, colZero.n_cols);

    expandVertically(&colZero, samples);
    expandVertically(&colOne, samples);

    arma::umat dataVisibleProbFalse = ( (1 - dataHidden) % colZero ) > arma::mat(samples, colZero.n_cols, arma::fill::randu);
    arma::umat dataVisibleProbTrue = ( dataHidden % colOne ) > arma::mat(samples, colOne.n_cols, arma::fill::randu);

    arma::umat dataVisible = dataVisibleProbFalse + dataVisibleProbTrue;

    return dataVisible;

}*/

std::shared_ptr<arma::umat> simulateVisibleData(arma::umat hiddenData, arma::mat distribution, std::mt19937 *engine) {

    //arma::umat visibleData(hiddenData);
    auto visibleData = std::make_shared<arma::umat>(hiddenData);
    
    visibleData->transform([&] (int val) {

        arma::colvec col = distribution.col(val);
        std::discrete_distribution<> dist(col.begin(), col.end());

        return dist(*engine);

    });

    return std::make_shared<arma::umat>(arma::trans(*visibleData));
}

arma::mat computeThetaHidden(arma::umat* hiddenData) {

    arma::umat histogram = arma::hist(arma::conv_to<::arma::rowvec>::from(*hiddenData), HIDDEN_STATES);
    arma::mat thetaHidden = arma::conv_to<arma::mat>::from(histogram);

    thetaHidden /= accu(thetaHidden);

    return thetaHidden;
}

arma::mat computeThetaVisible(arma::umat* dataHidden, arma::umat* dataVisible) {

    arma::mat thetaVisible = arma::mat(dataVisible->n_cols, 2, arma::fill::zeros);

    for (int i = 0; i < dataVisible->n_cols; ++i) {

        arma::umat visibleCol = arma::trans(dataVisible->col(i));

        thetaVisible.at(i, 0) = arma::accu(visibleCol % (1 - *dataHidden)) / (float) arma::accu(1 - *dataHidden);
        thetaVisible.at(i, 1) = arma::accu(visibleCol % *dataHidden) / (float) arma::accu(*dataHidden);

    }
    thetaVisible.transform( [] (double val) { return (std::isnan(val) ? double(0) : val); });

    return thetaVisible;
}

std::shared_ptr<arma::mat> computeThetaVisibleForNode(const arma::umat& hiddenData, const arma::umat& visibleData) {

    auto thetaVisible = std::make_shared<arma::mat>(VISIBLE_STATES, HIDDEN_STATES, arma::fill::zeros);

    for (int i = 0; i < hiddenData.n_cols; ++i) {
        ++((*thetaVisible)(visibleData(i), hiddenData(i)));
    }

    thetaVisible->each_col([] (arma::colvec& col) {
        double total = arma::accu(col);
        col /= total;
    });

    return thetaVisible;
}

std::shared_ptr<std::vector<arma::mat>> computeThetaVisible(arma::umat& hiddenData, arma::umat& visibleData) {

    auto thetaVisible = std::make_shared<std::vector<arma::mat>>();
    arma::mat convertedVisibleData = arma::conv_to<arma::mat>::from(visibleData);
    
    convertedVisibleData.each_col([&] (arma::colvec& col) {
        std::shared_ptr<arma::mat> nodeTheta = computeThetaVisibleForNode(hiddenData, arma::conv_to<arma::umat>::from(col));
        thetaVisible->push_back(*nodeTheta);
    });

    return thetaVisible;
}

arma::mat replaceAllValues(arma::umat* dataVisible, int thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible) {

    arma::mat replacedValues = arma::conv_to<arma::mat>::from(*dataVisible);

    for (int i = 0; i < replacedValues.n_cols; ++i) {
        auto col = replacedValues.col(i);
        arma::mat node = thetaVisible->at(i);

        col.transform([&] (double val) {
            return node(val, thetaHidden);
        });
    }

    return replacedValues;
}

arma::mat imputeHiddenNode(arma::umat* dataVisible, arma::mat thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible, bool generateNewData) {

    arma::mat probVis1 = replaceAllValues(dataVisible, 1, thetaVisible);
    arma::mat probVis1Unnorm = thetaHidden(1) * arma::prod(probVis1, 1);
    
    arma::mat denominator(probVis1Unnorm.n_rows, probVis1Unnorm.n_cols, arma::fill::zeros);
    arma::mat hidden;

    for (int i = 0; i < thetaHidden.n_cols; ++i) {

        arma::mat probVis = replaceAllValues(dataVisible, i, thetaVisible);
        arma::mat probVisUnnorm = thetaHidden(i) * arma::prod(probVis, 1);

        hidden = arma::join_rows(hidden, probVisUnnorm);
        denominator += probVisUnnorm;

    };

    /*hidden.each_col([&] (arma::colvec& col) {
        col /= denominator(0);
        col.transform([] (double val) {
            return (std::isnan(val) ? double(0) : val);
        });
    });*/

    if (generateNewData) {

        arma::mat hiddenData(1, hidden.n_rows);
        for (int i = 0; i < hidden.n_rows; ++i) {

            arma::rowvec row = hidden.row(i);
            row = row / accu(row); // Normalize

            std::discrete_distribution<> dist(row.begin(), row.end());
            hiddenData(0, i) = dist(engine);
        }

        hidden = hiddenData;

        /*arma::umat hiddenData;
        hidden.each_col([&] (const arma::colvec& col) {
            arma::umat data = col > arma::mat(hidden.n_rows, 1, arma::fill::randu);
            hiddenData = arma::join_rows(hiddenData, data);
        });

        hidden = arma::conv_to<arma::mat>::from(hiddenData);
        hidden = arma::mean(hidden);*/
    }
    /*arma::mat hidden = probVis1Unnorm / denominator;

    hidden.transform( [] (double val) { return (std::isnan(val) ? double(0) : val); });
    
    if (generateNewData) {
        hidden = arma::conv_to<arma::mat>::from(arma::trans(hidden > arma::mat(hidden.n_rows, hidden.n_cols, arma::fill::randu)));
    }*/

    return hidden;

}

arma::mat learn(arma::umat dataHidden, arma::umat dataVisible, int learningIterations) {

    arma::mat thetaHidden = computeThetaHidden(&dataHidden);
    std::shared_ptr<std::vector<arma::mat>> thetaVisible = computeThetaVisible(dataHidden, dataVisible);

    for (int i = 0; i < learningIterations; ++i) {

        dataHidden = arma::conv_to<arma::umat>::from(imputeHiddenNode(&dataVisible, thetaHidden, thetaVisible, true));
        //thetaHidden = imputeHiddenNode(&dataVisible, thetaHidden, thetaVisible, true);

        /*if (computeThetaHidden(&dataHidden)(1) < 0.5) {
            dataHidden = 1 - dataHidden;
        }*/

        thetaHidden = computeThetaHidden(&dataHidden);
        thetaVisible = computeThetaVisible(dataHidden, dataVisible);

    }
    return thetaHidden;
}
