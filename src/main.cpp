#include <iostream>
#include "../include/DataHandler.h"
#include "../include/Neuron.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"


int main() {
    std::string file_path = "../data.csv";
    char delim = ';';
    DataHandler dh;

    // Load CSV file
    dh.readCSV(file_path, delim);

    // Split the data into training, testing, and validation sets
    dh.TrainTestSplit();

    // Check if data was loaded correctly
    if (dh.getTrainingData().empty() || dh.getTestData().empty() || dh.getValidationData().empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return 1;
    }

    // Uncomment these lines for debugging purposes
    // dh.printDataArray();
    // dh.printTrainingData();
    // dh.printTestData();
    // dh.printValidationData();

    // Architecture variables
    double learningRate = 0.1;
    size_t outputSize = 1;                        // One for regression, more than one for classification
    std::vector<size_t> architecture = {5, 3, outputSize}; // Number of neurons for each layer
    int numEpochs = 10;

    // Ensure we have enough data to proceed
    if (dh.getTrainingData().empty()) {
        std::cerr << "Training data is empty. Cannot proceed with training." << std::endl;
        return 1;
    }

    // Setup and train the neural network
    try {
        NeuralNetwork net(architecture,                                             // Architecture
                          dh.getTrainingData().at(0).getFeatureVectorSize(),        // InputSize
                          outputSize,                                               // OutputSize
                          learningRate);                                            // LearningRate

        net.setTrainingData(dh.getTrainingData());
        net.setTestData(dh.getTestData());
        net.setValidationData(dh.getValidationData());
        net.train(numEpochs);                                                       // NumEpochs
        net.validate();
        net.test();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
