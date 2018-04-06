#ifndef TRAIN_HMM_H
#define TRAIN_HMM_H
#include "hmm.h"
#include <stdlib.h>
#include <string.h>
#include <string>
using namespace std;
double** alloc_2D_matrix(int m, int n);
void calculate_alpha(const HMM* hmm, const string& observe_seq, int length, double** alpha);
void calculate_beta(const HMM* hmm, const string& observe_seq, int length, double** beta);
void calculate_gamma(double** alpha, double** beta, int state_num, int length, double** gamma);
void calculate_epsilon(const HMM* hmm, double** alpha, double** beta, int length, const string& observe_seq,
    double*** epsilon);
void accumulate_gamma_epsilon(const HMM* hmm, double** gamma, double*** epsilon, const string& observe_seq, int length,
    double* gamma_init, double* gamma_sum, double** gamma_observe, double** epsilon_sum);
void reestimate_hmm(HMM* hmm, double* gamma_init, double* gamma_sum, double** gamma_observe,
    double** epsilon_sum, int nSamples);
#endif
