#include <iostream>
#include "vectorAdd.hh"

using namespace std;

int main(int argc, char** argv) {
    cout << "begin cudatest" << endl;

    cout << "-------------- test 50000 --------------" << endl;
    run_test(50000);
    cout << "----------------------------------------" << endl;

    cout << "-------------- test 500000 -------------" << endl;
    run_test(500000);
    cout << "----------------------------------------" << endl;

    cout << "end cudatest" << endl;
    return 0;
}