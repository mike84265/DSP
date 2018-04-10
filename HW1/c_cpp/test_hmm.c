#include "test_hmm.h"
#include <string.h>

double calculate_model(HMM* hmm, const char* observ)
{
    double delta[hmm->state_num];
    double prev[hmm->state_num];
    int observ_int = observ[0] - 'A';
    size_t length = strlen(observ);
    size_t t, i, j;
    for (i=0; i<hmm->state_num; ++i)
        prev[i] = hmm->initial[i] * hmm->observation[observ_int][i];
    for (t=1; t<length; ++t) {
        for (i=0; i<hmm->state_num; ++i) {
            delta[i] = 0;
            observ_int = observ[t] - 'A';
            for (j=0; j<hmm->state_num; ++j) {
                double value = prev[j] * hmm->transition[j][i] * hmm->observation[observ_int][i];
                if (value > delta[i]) delta[i] = value;
            }
        }
        for (i=0; i<hmm->state_num; ++i)
            prev[i] = delta[i];
    }
    double maxP = 0;
    for (i=0; i<hmm->state_num; ++i) {
        if (delta[i] > maxP)
            maxP = delta[i];
    }
    return maxP;
}
