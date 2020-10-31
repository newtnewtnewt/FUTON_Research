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
		double discountValue, const double gamma, const double* behaviorPolicyValues, const int bpvSize, 
		const double* optimalPolicyValues, const int opvSize, const double* rewardPolicyValues, const int rpvSize, 
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
	
int main() {
}