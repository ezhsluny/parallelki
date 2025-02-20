#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#define _USE_MATH_DEFINES

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif

using namespace std;
using namespace std::chrono;

int main() {
    auto start = high_resolution_clock::now();
    int size = pow(10, 7);
    vector<FLOAT_TYPE> values(size);

    FLOAT_TYPE arg = 0.0;
    FLOAT_TYPE step = 2 * M_PI / size;
    FLOAT_TYPE sum = 0;
    for (int i = 0; i < size; i++) {
        values.push_back(sin(arg));
        sum += values[i];
        arg += step;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << fixed << setprecision(10) << sum << endl;
    cout << duration.count() << endl;
    return 0;
}