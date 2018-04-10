#include "train_hmm.h"
#include <assert.h>
using namespace std;
double** alloc_2D_matrix(int m, int n)
{
    double** matrix = new double* [m];
    for (int i=0;i<m;++i)
        matrix[i] = new double[n];
    return matrix;
}

void calculate_alpha(const HMM* hmm, const string& observe_seq, int length,
    double** alpha)
{
    // alpha[t][i]
    // t: time
    // i: state
    int t, i, j;
    for (i=0; i<hmm->state_num; ++i) {
        int observ_num = observe_seq[0] - 'A';
        alpha[0][i] = hmm->initial[i] * hmm->observation[observ_num][i];
    }
    for (t=1; t<length; ++t) 
        for (j=0; j<hmm->state_num; ++j) {
            int observ_num = observe_seq[t] - 'A';
            assert(observ_num >= 0 && observ_num < 6);
            alpha[t][j] = 0;
            for (i=0; i<hmm->state_num; ++i)
                alpha[t][j] += (alpha[t-1][i] * hmm->transition[i][j]);
            alpha[t][j] *= hmm->observation[observ_num][j];
        }
}

void calculate_beta(const HMM* hmm, const string& observe_seq, int length,
    double** beta)
{
    // beta[t][i]
    // t: time
    // i: state
    int t, i, j;
    for (i=0; i<hmm->state_num; ++i)
        beta[length][i] = 1;
    for (t=length-2; t>=0; --t)
        for (j=0; j<hmm->state_num; ++j){
            int observ_num = observe_seq[t+1] - 'A';
            assert(observ_num >= 0 && observ_num < 6);
            beta[t][j] = 0;
            for (i=0; i<hmm->state_num; ++i)
                beta[t][j] += hmm->transition[i][j] * hmm->observation[observ_num][j] * beta[t+1][j];
        }
}

void calculate_gamma(double** alpha, double** beta, int state_num, int length,
    double** gamma)
{
    // gamma[t][i]
    // t: time
    // i: state
    double sum_t[state_num];
    int t,i,j;
    for (i=0; i<state_num; ++i)
        sum_t[i] = 0;
    for (t=0; t<length-1; ++t) {
        for (i=0; i<state_num; ++i) {
            gamma[t][i] = 0;
            for (j=0; j<state_num; ++j)
                gamma[t][i] += alpha[t][j] * beta[t][j];
            sum_t[i] += gamma[t][i];
        }
    }
    for (i=0; i<state_num; ++i) 
        gamma[length-1][i] = sum_t[i];
}

void calculate_epsilon(const HMM* hmm, double** alpha, double** beta, int length, const string& observe_seq,
    double*** epsilon)
{
    // epsilon[t][i][j]
    // t: time
    // i: state @ t
    // j: state @ t+1
    int t, i, j;
    for (i=0; i<hmm->state_num; ++i)
        for (int j=0; j<hmm->state_num; ++j)
            epsilon[length-1][i][j] = 0;
    for (t=0; t<length-1; ++t) {
        for (i=0; i<hmm->state_num; ++i)
            for (j=0; j<hmm->state_num; ++j) {
                int observ_num = observe_seq[t+1] - 'A';
                assert(observ_num >= 0 && observ_num < 6);
                double num = alpha[t][i] * hmm->transition[i][j] * hmm->observation[observ_num][j] * beta[t+1][j];
                double den = 0;
                for (int ii=0; ii<hmm->state_num; ++ii)
                    for (int jj=0; jj<hmm->state_num; ++jj)
                        den += alpha[t][ii] * hmm->transition[ii][jj] * hmm->observation[observ_num][jj] * beta[t+1][jj];
                epsilon[t][i][j] = num / den;
                epsilon[length-1][i][j] += num / den; // epsilon[T-1] stores the sum over t=0~T-2
            }
    }
}

void accumulate_gamma_epsilon(const HMM* hmm, double** gamma, double*** epsilon, const string& observe_seq, int length,
    double* gamma_init, double* gamma_sum, double** gamma_observe, double** epsilon_sum)
{
    // gamma_init[i]: sum of initial gamma at t=0 of state i
    // gamma_sum[i]: sum of gamma among t=0~T-2 of state i
    // gamma_observe[ob][i]: sum of gamma of state i given observation
    // epsilon_sum[i][j]: sum of epsilon from state i to state j
    int t, i, j;
    for (i=0; i<hmm->state_num; ++i)
        gamma_init[i] += gamma[0][i];
    for (t=0; t<length-1; ++t) {
        for (i=0; i<hmm->state_num; ++i) {
            gamma_sum[i] += gamma[t][i];
            int observ_int = observe_seq[t] - 'A';
            assert(observ_int >= 0 && observ_int < 6);
            gamma_observe[observ_int][i] += gamma[t][i];
        }
    }
    for (i=0; i<hmm->state_num; ++i)
        for (j=0; j<hmm->state_num; ++j)
            epsilon_sum[i][j] += epsilon[length-1][i][j];
}

void reestimate_hmm(HMM* hmm, double* gamma_init, double* gamma_sum, double** gamma_observe,
    double** epsilon_sum, int nSamples)
{
    int i, j, k;
    for (i=0; i<hmm->state_num; ++i)
        hmm->initial[i] = gamma_init[i] / nSamples;

    for (i=0; i<hmm->state_num; ++i)
        for (j=0; j<hmm->state_num; ++j)
            hmm->transition[i][j] = epsilon_sum[i][j] / gamma_sum[i];
    
    for (k=0; k<hmm->observ_num; ++k)
        for (i=0; i<hmm->state_num; ++i)
            hmm->observation[k][i] = gamma_observe[k][i] / gamma_sum[i];
}