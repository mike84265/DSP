#include "hmm.h"
#include <string.h>
#include <stdlib.h>
double** alloc_2D_matrix(int m, int n)
{
    double** matrix = (double**)malloc(sizeof(double*) * m);
    for (int i=0;i<m;++i)
        matrix[i] = (double*)malloc(sizeof(double) * n);
    return matrix;
}
double** calculate_alpha(const HMM* hmm, const char* observe_seq, int length)
{
    // alpha[t][i]
    // t: time
    // i: state
    double** alpha = alloc_2D_matrix(length, hmm->state_num);
    for (int i=0; i<hmm->state_num; ++i)
        alpha[0][i] = hmm->initial[i] * hmm->observation[observe_seq[0]][i];
    for (int t=1; t<length; ++t) 
        for (int j=0; j<hmm->state_num; ++j) {
            int observ_num = observe_seq[t] - 'A';
            alpha[t][i] = 0;
            for (int i=0; i<hmm->state_num ++i)
                alpha[t][i] += (alpha[t-1][i] * hmm->transition[i][j]);
            alpha[t][i] *= hmm->observation[observ_num][j];
        }
    return alpha;
}

double** calculate_beta(const HMM* hmm, const char* observe_seq, int length)
{
    // beta[t][i]
    // t: time
    // i: state
    double** beta = alloc_2D_matrix(length, hmm->state_num);
    for (int i=0; i<hmm->state_num; ++i)
        beta[length][i] = 1;
    for (int t=length-2; t>=0; --t)
        for (int j=0; j<hmm->state_num; ++j){
            int observ_num = observe_seq[t+1] - 'A';
            beta[t][j] = 0;
            for (int i=0; i<hmm->state_num; ++i)
                beta[t][j] += hmm->transition[i][j] * hmm->observation[observ_num][j] * beta[t+1][j];
        }
    return beta; 
}

double** calculate_gamma(double** alpha, double** beta, int state_num, int length)
{
    // gamma[t][i]
    // t: time
    // i: state
    double** gamma = alloc_2D_matrix(length, state_num);
    double sum_t[state_num];
    for (int i=0; i<state_num; ++i)
        sum_t[i] = 0;
    for (int t=0; t<length-1; ++t) {
        for (int i=0; i<state_num; ++i) {
            gamma[t][i] = 0;
            for (int j=0; j<state_num; ++j)
                gamma[t][i] += alpha[t][j] * beta[t][j];
            sum_t[i] += gamma[t][i];
        }
    }
    for (int i=0; i<state_num; ++i) 
        gamma[length-1][i] = sum_t[i];
    return gamma;
}

double*** calculate_epsilon(const HMM* hmm, double** alpha, double** beta, int length, const char* observe_seq)
{
    // epsilon[t][i][j]
    // t: time
    // i: state @ t
    // j: state @ t+1
    double*** epsilon = (double***)malloc(sizeof(double**) * length);
    double sum_t[hmm->state_num][hmm->state_num];
    for (int i=0; i<hmm->state_num; ++i)
        for (int j=0; j<hmm->state_num; ++j)
            sum_t[i][j] = 0;
    for (int i=0; i<length; ++i)
        epsilon[i] = alloc_2D_matrix(hmm->state_num, hmm->state_num);
    for (int t=0; t<length-1; ++t) {
        for (int i=0; i<hmm->state_num; ++i)
            for (int j=0; j<hmm->state_num; ++j) {
                int observ_num = observe_seq[t+1] - 'A';
                double num = alpha[t][i] * hmm->transition[i][j] * observation[observ_num][j] * beta[t+1][j];
                double den = 0;
                for (int ii=0; ii<hmm->state_num; ++ii)
                    for (int jj=0; jj<hmm->state_num; ++jj)
                        den += alpha[t][ii] * hmm->transition[ii][jj] * observation[observ_num][jj] * beta[t+1][jj];
                epsilon[t][i][j] = num / den;
                sum_t[i][j] += num / den;
            }
    }
}

void reestimate_hmm(HMM* hmm, double** gamma_seq, double*** epsilon_seq, int nSamples)
{
    for (int i=0; i<hmm->state_num; ++i) {
        hmm->initial[i] = 0;
        for (int k=0; k<nSamples; ++k)
            hmm->initial[i] += gamma_seq[k][i];
        hmm->initial[i] /= nSamples;
    }

    for (int i=0; i<hmm->state_num; ++i)
        for (int j=0; j<hmm->state_num; ++j) {
            double num=0, den=0;
            for (int k=0; k<nSamples; ++k){
                num += epsilon_seq[k][i][j];
                den += gamma_seq[k][i];
            }
            hmm->transition[i][j] = num / den;
        }
    
    for (int i=0; i<hmm->state_num; ++i)
        for (int k=0; k<nSamples; ++k) {
            
        }
}
