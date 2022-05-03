#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>

#define PRINT_TIME(code) do { \
    auto start = system_clock::now(); \
    code \
    auto end   = system_clock::now(); \
    auto duration = duration_cast<microseconds>(end - start); \
    cout << "time spent: " << double(duration.count()) << "us" << endl; \
} while(0)

using namespace std;

using namespace chrono;

using vec = vector<int>;

const int scale[] = {256, 512, 1024, 2048};
const string data_path("./data/");

void Gemm(const int &inSize, vec &inA, vec &inB, vec &outC)
{
    register const int size = inSize;
    register const int* a = inA.data();
    register const int* b = inB.data();
    register int* c = outC.data();
    for(register int i = 0; i < size; i++)
    {
        for(register int j = 0; j < size; j++)
        {
            register int sum = 0;
            for(register int k = 0; k < size; k++)
               sum += a[i * size + k] * b[k * size + j];
            c[i * size + j] = sum;
        }
    }
}

void CheckResult(const vec &c, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = c.size();
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        assert(c[i] == res_i);
    }
    file_result.close();
}

// c = a * b
void Benchmark(const int &size) {
    const int nelems = size * size;
    const string a_path(data_path+to_string(size)+"/a");
    const string b_path(data_path+to_string(size)+"/b");
    const string result_path(data_path+to_string(size)+"/result");
    ifstream file_a(a_path);
    ifstream file_b(b_path);

    vec a(nelems, 0);
    vec b(nelems, 0);
    vec c(nelems, 0);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
       Gemm(size, a, b, c);
    );

    CheckResult(c, result_path);

    file_a.close();
    file_b.close();
}

int main() {
    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}
