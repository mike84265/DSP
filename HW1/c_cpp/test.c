#include "hmm.h"
#include "test_hmm.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_MODEL 64
int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s modellist testing_data result\n", argv[0]);
        exit(1);
    }

    HMM hmm[MAX_MODEL];
    int numModels = load_models(argv[1], hmm, MAX_MODEL);
    FILE* fin = open_or_die(argv[2], "r");
    FILE* fout = open_or_die(argv[3], "w");
    #ifdef DEBUG
    FILE* log = open_or_die("log.txt", "w");
    #endif

    double maxP = 0;
    int maxI;
    char buffer[256];

    while ((fscanf(fin, "%s", buffer)) != EOF) {
        maxP = 0;
        maxI = -1;
        for (int i=0; i<numModels; ++i) {
            double val = calculate_model(&hmm[i], buffer);
            if (val > maxP) {
                maxP = val;
                maxI = i;
            }
            #ifdef DEBUG
            fprintf(log, "%s %E\n", hmm[i].model_name, val);
            #endif
        }
        fprintf(fout, "%s %E\n", hmm[maxI].model_name, maxP);
    }
    fclose(fin);
    fclose(fout);
    #ifdef DEBUG
    fclose(log);
    #endif
}
