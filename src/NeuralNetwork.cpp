#include "../include/NeuralNetwork.h"

// Create architecture of ANN in order to connect each layer from InputLayer to outputLayer
NeuralNetwork::NeuralNetwork(std::vector<size_t> architecture, size_t inputSize, size_t outputLayer,
                             double learningRate) : learningRate(learningRate), beta1(0.9), beta2(0.999), epsilon(1e-8), t(0)
{
    if (architecture.empty()) {
        throw std::invalid_argument("Architecture cannot be empty.");
    }
    layers.reserve(architecture.size());
    for (size_t i = 0; i < architecture.size(); i++) {
        size_t layerInputSize = (i == 0) ? inputSize : layers.back()->neurons.size();
        layers.emplace_back(std::make_unique<Dense>(layerInputSize,layerInputSize, architecture[i]));
    }
    layers.emplace_back(std::make_unique<Linear>(layers.back()->neurons.size(), layers.back()->neurons.size(), outputLayer));
}

NeuralNetwork::~NeuralNetwork() = default;



std::vector<double> NeuralNetwork::propagateThroughLayer(std::unique_ptr<Layer>& layer, const std::vector<double>& inputs,
                                                         bool isOutputLayer)
{
    std::vector<double> newInputs;
    newInputs.reserve(layer->neurons.size());

    for (auto& neuron : layer->neurons) {
        double activationValue = activation(neuron->weights, inputs);
        neuron->output = isOutputLayer ? linear(activationValue) : ReLU(activationValue);
        newInputs.push_back(neuron->output);
    }

    return newInputs;
}

double NeuralNetwork::FeedForwardPropagation(const Data& data)
{
    std::vector<double> inputs = data.getFeatureVector();
    for (size_t i = 0; i < layers.size(); ++i) {
        inputs = propagateThroughLayer(layers[i], inputs, (i == layers.size() - 1));
    }
    return inputs[0];
}

double NeuralNetwork::activation(const std::vector<double>& weights, const std::vector<double>& input)
{
    if (weights.size() != input.size() + 1) {
        throw std::invalid_argument("The size of weights must be equal to the size of input plus one for the bias term.");
    }

    double activation = weights.back();
    for (size_t i = 0; i < input.size(); ++i) {
        activation += weights[i] * input[i];
    }
    return activation;
}

inline double NeuralNetwork::ReLU(double activation)
{
    return std::max(activation, 0.0);
}

inline double NeuralNetwork::linear(double activation)
{
    return activation;
}

inline double NeuralNetwork::derivativeReLu(double output)
{
    return (output > 0.0) ? 1.0 : 0.0;
}

inline double NeuralNetwork::derivativeLinear(double output)
{
    return 1.0;
}

void NeuralNetwork::backPropagation(const Data& data)
{
    int lastLayer = (int)layers.size() - 1;
    for (int i = lastLayer; i > -1; i--) {
        std::vector<double> errors;
        std::unique_ptr<Layer>& layer = layers.at(i);
        int sizeNeurons = (int)layer->neurons.size();
        errors.reserve(sizeNeurons);
        if (i != lastLayer) {
            for (int j = 0; j < sizeNeurons; j++) {
                double error = 0.0;
                for (std::unique_ptr<Neuron>& n : layers.at(i + 1)->neurons) {
                    double tempError = n->weights.at(j) * n->delta;
                    error += tempError;
                }
                errors.push_back(error);
            }
        } else {
            for (std::unique_ptr<Neuron>& n : layer->neurons) {
                double error = n->output - data.getTarget();
                errors.push_back(error); // expected - actual
            }
        }
        for (int j = 0; j < sizeNeurons; j++) { // iterate through neurons of layer
            std::unique_ptr<Neuron>& n = layer->neurons.at(j); // gradient/derivative part.
            if (i != lastLayer) {
                double errorDelta = errors.at(j) * this->derivativeReLu(n->output);
                n->delta = errorDelta;
            } else {
                double errorDelta = errors.at(j) * this->derivativeLinear(n->output); // 1
                n->delta = errorDelta;
            }
        }
    }
}

void NeuralNetwork::updateWeights(const Data& data) {    // Adam optimizer
    std::vector<double> inputs = data.getFeatureVector();
    t++;
    for (size_t i = 0; i < layers.size(); i++) {
        if (i != 0) {
            for (auto& n : layers.at(i - 1)->neurons) {
                inputs.push_back(n->output);
            }
        }
        for (auto& n : layers.at(i)->neurons) {
            for (size_t j = 0; j < inputs.size(); j++) {
                double g = n->delta * inputs[j]; // Gradient
                n->m[j] = beta1 * n->m[j] + (1 - beta1) * g;
                n->v[j] = beta2 * n->v[j] + (1 - beta2) * g * g;

                double m_hat = n->m[j] / (1 - pow(beta1, t));
                double v_hat = n->v[j] / (1 - pow(beta2, t));

                n->weights[j] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
            }
            double g_bias = n->delta; // Gradient for bias
            n->m.back() = beta1 * n->m.back() + (1 - beta1) * g_bias;
            n->v.back() = beta2 * n->v.back() + (1 - beta2) * g_bias * g_bias;

            double m_hat_bias = n->m.back() / (1 - pow(beta1, t));
            double v_hat_bias = n->v.back() / (1 - pow(beta2, t));

            n->weights.back() -= learningRate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
        }
        inputs.clear();
    }
}

double NeuralNetwork::predict(const Data& data)
{
    double output = this->FeedForwardPropagation(data);
    return output;
}

void NeuralNetwork::train(int numEpochs)
{
    for (int i = 0; i < numEpochs; i++) {
        std::vector<double> predicted;
        std::vector<double> actual;
        predicted.reserve(trainingData.size());
        actual.reserve(trainingData.size());
        double total_loss = 0.0;
        for (const Data& trainData : this->trainingData) {
            double y_predicted = this->FeedForwardPropagation(trainData);
            double y_expected = trainData.getTarget();
            double loss = 0.5 * std::pow(y_predicted - y_expected, 2); // Mean squared error
            total_loss += loss;
            predicted.push_back(y_predicted);
            actual.push_back(y_expected);
            backPropagation(trainData);
            updateWeights(trainData);
        }
        double trainR2score = r2_score(actual, predicted);
        double trainMAEscore = MAE_score(actual, predicted);
        std::cout << "Epoch " << i << " | loss(MSE): " << total_loss << " | R2 score: " << trainR2score
                  << " | MAE score: " << trainMAEscore << "\n";
    }
    std::cout << "-------------------------------------------------------------------------\n";
}

void NeuralNetwork::test()
{
    std::vector<double> predicted;
    std::vector<double> actual;
    predicted.reserve(testData.size());
    actual.reserve(testData.size());
    for (const Data& testDatum : this->testData) {
        double PredictionTemp = this->NeuralNetwork::predict(testDatum);
        predicted.push_back(PredictionTemp);
        actual.push_back(testDatum.getTarget());
    }
    double testR2score = r2_score(actual, predicted);
    double testMAEscore = MAE_score(actual, predicted);

    std::cout << "R2 score test dataset: " << testR2score << "\n";
    std::cout << "MAE score test dataset: " << testMAEscore << "\n";
    std::cout << "-------------------------------------------------\n";
}

void NeuralNetwork::validate()
{
    std::vector<double> predicted;
    std::vector<double> actual;
    predicted.reserve(validationData.size());
    actual.reserve(validationData.size());
    for (const Data& valData : this->validationData) {
        double PredictionTemp = this->predict(valData);
        predicted.push_back(PredictionTemp);
        actual.push_back(valData.getTarget());
    }
    double validationR2score = r2_score(actual, predicted);
    double validationMAEscore = MAE_score(actual, predicted);

    std::cout << "R2 score validation dataset: " << validationR2score << "\n";
    std::cout << "MAE score validation dataset: " << validationMAEscore << "\n";
    std::cout << "-------------------------------------------------\n";
}
