#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
using namespace std;
int main(int argc, char** argv)
{
    ifstream f1(argv[1], ios::in), f2(argv[2], ios::in);
    string s1, s2;
    int correct = 0, total = 0;
    while (1) {
        getline(f1, s1);
        getline(f2, s2);
        if (f1.eof() or f2.eof())
            break;
        size_t n = s1.find_first_of(' ');
        s1 = s1.substr(0, n);
        n = s2.find_first_of(' ');
        s2 = s2.substr(0, n);
        if (s1 == s2)
            ++correct;
        ++total;
    }
    printf("accuracy = %d/%d = %.4f\%\n", correct, total, (double)correct / total * 100);
}
