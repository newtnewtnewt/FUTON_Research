#include <cstdlib>
#include <cmath>
#include <iostream>
//#include <vector>

/*
* performanceEvaluation is a helper method to optimize floating 
* point calls in python
* It is equivalent to the already documented python code located in the 
* offpolicy_eval_wis method of Full Q-Learning Script
* http://www.mingw.org/wiki/sampledll
* g++ -c -fPIC wis_speedup.cpp -o wis_speedup.o
* g++ -o libwis.dll wis_speedup.o -static -Wl,--out-implib,libwis.dll.a
*/

extern "C" __declspec( dllexport ) double* performanceEvaluation(double currentTrialEval, double rhoValue, 
		double discountValue, double gamma, double* behaviorPolicyValues, int bpvSize, 
		double* optimalPolicyValues, int opvSize, double* rewardPolicyValues, int rpvSize, 
		int startValue, int endValue){
			for(int i = startValue; i < endValue; i++) {
				rhoValue = rhoValue * (*(behaviorPolicyValues + i) / *(optimalPolicyValues + i));
				discountValue = discountValue * gamma;
				currentTrialEval = currentTrialEval + discountValue * (*(rewardPolicyValues + i + 1));
				// Strangely, this does not work!
				//currentTrialEval = std::fma(discountValue, *(rewardPolicyValues + i + 1), currentTrialEval);
			}
			static double vecPointer[2]; 
			vecPointer[0] = rhoValue;
			vecPointer[1] = currentTrialEval;
			return vecPointer;
		}

extern "C" __declspec( dllexport ) double check() { return 2.0; }

extern "C" __declspec( dllexport ) double rToSender(double* start, int size) {
	double total = 0.0;
	for(int i = 0; i < size; i++) {
		total += *(start + i);
	}
	return total;
}
extern "C" __declspec( dllexport ) double* pointReturn(double a) { 
	static double x = 2.0 + a;
	static double* y = &x;
	return y;
}
	
int main() {
	/*
	double x = 2.0;
	double y = 3.0;
	double* z = &x;
	double* zz = &y;
	double* fin = pointReturn(z, zz);
	std::cout << *fin << std::endl;
	*/
	/*
	double x[2] = {1.0, 2.0};
	std::cout << rToSender(x, 2) << std::endl;
	*/
	/*
	double currentTrialEval = 0.0;
	double rho_value = 1.0; 
	double discount_value = 1/0.99;
    double gamma = 0.99;
	std::vector<double> behaviorPolicyValues = {20, 30, 40, 50, 60};
	std::vector<double> optimalPolicyValues = {40, 70, 20, 90, 100};
	std::vector<double> rewardPolicyValues = {5, 10, 15, 20, 25};
	double* behPointer = &behaviorPolicyValues[0];
	double* optPointer = &optimalPolicyValues[0];
	double* rewPointer = &rewardPolicyValues[0];	
	double* returnVal = performanceEvaluation(currentTrialEval, rho_value, discount_value, gamma, 
	behPointer, 5, optPointer, 5, rewPointer, 5, 0, 5);
	std::cout << "This is the first value I have " << *returnVal << std::endl;
	std::cout << "This is the second value I have " << *(returnVal + 1) << std::endl;
	return 0;
	*/
}