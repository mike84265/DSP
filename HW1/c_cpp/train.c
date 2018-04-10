#include "train_hmm.h"
#include "hmm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s iteration model_init seq_model model_out", argv[0]);
        exit(1);
    }
    int iteration = atoi(argv[1]);
    HMM hmm;
    loadHMM(&hmm, argv[2]);

    int fd = open(argv[3], O_RDONLY);
    if (fd < 0) {
        perror(argv[3]);
        exit(1);
    }
    size_t file_length = lseek(fd, 0, SEEK_END);
    char* pMap = mmap(0, file_length, PROT_READ, MAP_PRIVATE, fd, 0);
    char* ptr = strchr(pMap, '\n');
    char* prev = pMap;

    size_t length = ptr - pMap;
    size_t lineNo = 0;
    int state_num = hmm.state_num;
    double** alpha = alloc_2D_matrix(length, state_num);
    double** beta = alloc_2D_matrix(length, state_num);
    double** gamma = alloc_2D_matrix(length, state_num);
    double*** epsilon = (double***)malloc(sizeof(double**) * length);
    for (int i=0; i<length; ++i)
        epsilon[i] = alloc_2D_matrix(state_num, state_num);
    double* gamma_init = (double*)malloc(sizeof(double) * state_num);
    double* gamma_sum = (double*)malloc(sizeof(double) * state_num);
    double** gamma_observe = alloc_2D_matrix(hmm.observ_num, state_num);
    double** epsilon_sum = alloc_2D_matrix(state_num, state_num);


    for (int it=0; it<iteration; ++it) {
        // initialize sums
        for (int i=0; i<state_num; ++i) {
            gamma_init[i] = 0;
            gamma_sum[i] = 0;
            for (int j=0; j<state_num; ++j)
                epsilon_sum[i][j] = 0;
        }
        for (int i=0; i<hmm.observ_num; ++i)
            for (int j=0; j<state_num; ++j)
                gamma_observe[i][j] = 0;
        ptr = strchr(pMap, '\n');
        prev = pMap;
        lineNo = 0;

        while(ptr != NULL) { 
            calculate_alpha(&hmm, prev, length, alpha);
            calculate_beta(&hmm, prev, length, beta);
            calculate_gamma(alpha, beta, state_num, length, gamma);
            calculate_epsilon(&hmm, alpha, beta, length, prev, epsilon);
            accumulate_gamma_epsilon(&hmm, gamma, epsilon, prev, length, 
                gamma_init, gamma_sum, gamma_observe, epsilon_sum);

            ++lineNo;
            prev = ptr+1;
            ptr = strchr(ptr+1, '\n');
        }
        reestimate_hmm(&hmm, gamma_init, gamma_sum, gamma_observe, epsilon_sum, lineNo);
        #ifdef DEBUG
        system("clear");
        printf("At iteration %d/%d: \n", it, iteration);
        dumpHMM(stdout, &hmm);
        #endif
    }

    close(fd);
    munmap(pMap, file_length);
    
    FILE* fp = fopen(argv[4], "w");
    dumpHMM(fp, &hmm);
    fclose(fp);

    // Free Dynamic arrays
    for (int i=0; i<length; ++i) {
        free(alpha[i]);
        free(beta[i]);
        free(gamma[i]);
        for (int j=0; j<state_num; ++j)
            free(epsilon[i][j]);
        free(epsilon[i]);
    }
    for (int i=0; i<hmm.observ_num; ++i)
        free(gamma_observe[i]);
    for (int i=0; i<state_num; ++i)
        free(epsilon_sum[i]);
    free(alpha);
    free(beta);
    free(gamma);
    free(epsilon);
    free(gamma_init);
    free(gamma_sum);
    free(gamma_observe);
    free(epsilon_sum);
}
