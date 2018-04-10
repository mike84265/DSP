#include "hmm.h"
#include "test_hmm.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
using namespace std;

#define MAX_MODEL 64
int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s modellist testing_data, result\n", argv[0]);
        exit(1);
    }

    HMM hmm[MAX_MODEL];
    int numModels = load_models(argv[1], hmm, MAX_MODEL);
    ifstream ifs(argv[2], ios::in);
    double maxP = 0;
    string buffer;
    int maxI;

    if (!ifs) {
        perror(argv[2]);
        exit(1);
    }
    ofstream ofs(argv[3], ios::out);
    while (!ifs.eof()) {
        ifs >> buffer;
        for (int i=0; i<numModels; ++i) {
            double val = calculate_model(&hmm[i], buffer);
            if (val > maxP) {
                maxP = val;
                maxI = i;
            }
        }
        ofs << hmm[maxI].model_name << endl;
    }
}
