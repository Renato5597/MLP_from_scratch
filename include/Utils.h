#ifndef ML_MLP_UTILS_H
#define ML_MLP_UTILS_H


#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

double Mean(const std::vector<double>&);
double RSS(const std::vector<double>&, const std::vector<double>& );       //residual sum of squares
double TSS(const std::vector<double>&);                              //total sum of squares,
double r2_score(const std::vector<double>&, const std::vector<double>&);   //coefficient of determination
double MAE_score(const std::vector<double>&, const std::vector<double>&);  //mean absolute error
double MSE_score(const std::vector<double>&, const std::vector<double>&);   //mean_squared_error

#endif //ML_MLP_UTILS_H
