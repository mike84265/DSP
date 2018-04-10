#include "train_hmm.h"
#include "hmm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;
int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s iteration model_init seq_model model_out", argv[0]);
        exit(1);
    }
    int iteration = atoi(argv[1]);
    HMM hmm;
    loadHMM(&hmm, argv[2]);
    ifstream ifs_seq(argv[3], ios::in);
    if (!ifs_seq) {
        perror(argv[3]);
        exit(1);
    }
    vector<string> v_seq;
    string buffer;
    while (!ifs_seq.eof()) {
        ifs_seq >> buffer;
        v_seq.push_back(buffer);
    }
    ifs_seq.close();

    size_t length = v_seq[0].size();
    int state_num = hmm.state_num;
    double** alpha = alloc_2D_matrix(length, state_num);
    double** beta = alloc_2D_matrix(length, state_num);
    double** gamma = alloc_2D_matrix(length, state_num);
    double*** epsilon = new double** [length];
    for (int i=0; i<length; ++i)
        epsilon[i] = alloc_2D_matrix(state_num, state_num);
    double* gamma_init = new double [state_num];
    double* gamma_sum = new double [state_num];
    double** gamma_observe = alloc_2D_matrix(hmm.observ_num, state_num);
    double** epsilon_sum = alloc_2D_matrix(state_num, state_num);


    for (int it=0; it<iteration; ++it) {

        for (int i=0; i<state_num; ++i) {
            gamma_init[i] = 0;
            gamma_sum[i] = 0;
            for (int j=0; j<state_num; ++j)
                epsilon[i][j] = 0;
        }

        for (int i=0; i<hmm.observ_num; ++i)
            for (int j=0; j<state_num; ++j)
                gamma_observe[i][j] = 0;

        for (int s=0; s<v_seq.size(); ++s) {
            calculate_alpha(&hmm, v_seq[s], length, alpha);
            calculate_beta(&hmm, v_seq[s], length, beta);
            calculate_gamma(alpha, beta, state_num, length, gamma);
            calculate_epsilon(&hmm, alpha, beta, length, v_seq[s], epsilon);
            accumulate_gamma_epsilon(&hmm, gamma, epsilon, v_seq[s], length, 
                gamma_init, gamma_sum, gamma_observe, epsilon_sum);
            #ifdef DEBUG
            system("clear");
            printf("At iteration %d/%d, sequence %d/%d ...\n", it, iteration, s, v_seq.size());
            #endif
        }
        reestimate_hmm(&hmm, gamma_init, gamma_sum, gamma_observe, epsilon_sum, v_seq.size());
    }
    
    FILE* fp = fopen(argv[4], "w");
    dumpHMM(fp, &hmm);
    fclose(fp);

    // Free Dynamic arrays
    for (int i=0; i<length; ++i) {
        delete[] alpha[i];
        delete[] beta[i];
        delete[] gamma[i];
        for (int j=0; j<state_num; ++j)
            delete[] epsilon[i][j];
        delete[] epsilon[i];
    }
    for (int i=0; i<hmm.observ_num; ++i)
        delete[] gamma_observe[i];
    for (int i=0; i<state_num; ++i)
        delete[] epsilon_sum[i];
    delete[] alpha;
    delete[] beta;
    delete[] gamma;
    delete[] epsilon;
    delete[] gamma_init;
    delete[] gamma_sum;
    delete[] gamma_observe;
    delete[] epsilon_sum;
}
