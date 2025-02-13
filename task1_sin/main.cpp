#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#define _USE_MATH_DEFINES

using namespace std;
using namespace std::chrono;

int main() {
    auto start = high_resolution_clock::now();
    int size = pow(10, 7);
    vector<int> values(size);

    float arg = 0.0;
    float step = 2 * M_PI / size;
    float sum = 0;
    for (int i = 0; i < size; i++) {
        values.push_back(sin(arg));
        sum += values[i];
        arg += step;
    }

    
    // for (int i = 0; i < size; i++) {
    //     sum += values[i];
    // }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << fixed << setprecision(10) << sum << endl;
    cout << duration.count() << endl;
    return 0;
}