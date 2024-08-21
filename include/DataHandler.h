#ifndef ML_MLP_DATAHANDLER_H
#define ML_MLP_DATAHANDLER_H

#include "../include/Data.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>

class DataHandler
{
private:
    std::vector<Data> dataArray;  //all csv data
    std::vector<Data> trainingData;
    std::vector<Data> testData;
    std::vector<Data> validationData;
public:
    const size_t TRAIN_SET = 70;
    const size_t TEST_SET = 20;
    const size_t VALIDATION_SET = 10;

    DataHandler() noexcept;
    ~DataHandler() noexcept;
    void readCSV(const std::string&, const char&);
    void TrainTestSplit();

    //print
    void printDataArray() const;
    void printTrainingData() const;
    void printTestData() const;
    void printValidationData() const;

    //getters
    [[nodiscard]] const std::vector<Data>& getTrainingData() const;
    [[nodiscard]] const std::vector<Data>& getTestData() const;
    [[nodiscard]] const std::vector<Data>& getValidationData() const;
};

#endif //ML_MLP_DATAHANDLER_H
