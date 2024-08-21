#include <utility>
#include "../include/CommonData.h"

void CommonData::setTrainingData(std::vector<Data> vect1)
{
    trainingData = std::move(vect1);
}
void CommonData::setTestData(std::vector<Data> vect2)
{
    testData = std::move(vect2);
}
void CommonData::setValidationData(std::vector<Data> vect3)
{
    validationData = std::move(vect3);
}
