#ifndef ML_MLP_COMMONDATA_H
#define ML_MLP_COMMONDATA_H


#include "../include/Data.h"
#include <vector>
class CommonData
{
protected:
    std::vector<Data> trainingData;
    std::vector<Data> testData;
    std::vector<Data> validationData;
public:
    void setTrainingData(std::vector<Data>);
    void setTestData(std::vector<Data>);
    void setValidationData(std::vector<Data>);
};


#endif //ML_MLP_COMMONDATA_H
