/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *  Author: ecesar
 *  changes: Anna Sikora 2021-2022
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge (mirar descripció del format al comentari de readImg) de 32x32 (un total de 1024
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 117
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrament caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 10 neurones ja que l'entrenem per reconéixer 10
 *      patrons ['0'..'9']).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 1934 amb els dígits '0'..'9', escrits a mà). Un cop ha calculat
 * 	    els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconèixament, on llegeix 946 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quin dígit es tracta.
 *
 *  Darrera modificació: gener 2019. Ara l'aprenentatge fa servir la tècnica dels mini-batches
 */

/*******************************************************************************
 *    Aquest programa és una adaptació del fet per  JOHN BULLINARIA
 *    ( http://www.cs.bham.ac.uk/~jxb/NN/nn.html):
 *
 *    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include <limits.h>
#include <sys/time.h>
#include "common.h"
// #include "/usr/include/x86_64-linux-gnu/mpich/mpi.h"
#include <mpi.h>

#define MASTER 0
#define ERROR_MULTIPLICATION 1
#define DISPLAY_OUTPUT 0

int numberOfProcess, rank;

int total;
int seed = 50;

int rando()
{
    seed = (214013 * seed + 2531011);
    return seed >> 16;
}

float frando()
{
    return (rando() / 65536.0f);
}

void freeTSet(int np, char **tset)
{
    for (int i = 0; i < np; i++)
        free(tset[i]);
    free(tset);
}

void trainN()
{

    char **tSet;

    float DeltaWeightIH[NUMHID][NUMIN], DeltaWeightHO[NUMOUT][NUMHID];
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
    int ranpat[NUMPAT];
    float Hidden[NUMHID], Output[NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
    float SumO, SumH, SumDOW;

    // Repartir la feina
    int batchStartingPoint, batchEndingPoint;

    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL)
    {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    for (int i = 0; i < NUMHID; i++)
        for (int j = 0; j < NUMIN; j++)
        {
            if (rank == MASTER)
                WeightIH[i][j] = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightIH[i][j] = 0.0;
        }

    for (int i = 0; i < NUMOUT; i++)
        for (int j = 0; j < NUMHID; j++)
        {
            if (rank == MASTER)
                WeightHO[i][j] = 2.0 * (frando() + 0.01) * smallwt;
            DeltaWeightHO[i][j] = 0.0;
        }

    MPI_Bcast(WeightIH, NUMHID * NUMIN, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(WeightHO, NUMOUT * NUMHID, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    for (int epoch = 0; epoch < 1000000; epoch++)
    {

        if (rank == MASTER)
        {
            for (int p = 0; p < NUMPAT; p++)
                ranpat[p] = p;
            for (int p = 0; p < NUMPAT; p++)
            {
                int x = rando();
                int np = (x * x) % NUMPAT;
                int op = ranpat[p];
                ranpat[p] = ranpat[np];
                ranpat[np] = op;
            }
        }

        MPI_Bcast(ranpat, NUMPAT, MPI_INT, MASTER, MPI_COMM_WORLD);

        Error = BError = 0.0;

        if (rank == MASTER)
        {
            printf(".");
            fflush(stdout);
        }

        // Divide workload accross all nodes /!\ If not divisible by the number of nodes /!
        // If not divisible we give the last iterations to the last node...
        int startingIndex, endingIndex;
        int work = floor((NUMPAT / BSIZE) / numberOfProcess);
        startingIndex = rank * work;
        endingIndex = (rank + 1) * work;
        if (rank == numberOfProcess - 1)
            endingIndex = NUMPAT / BSIZE;
        // printf("Start : %d, Finish %d\t process %d\n", startingIndex, endingIndex, rank);

        for (int nb = startingIndex; nb < endingIndex; nb++)
        {
            BError = 0.0;
            for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++)
            {

                int p = ranpat[np];
                for (int j = 0; j < NUMHID; j++)
                {
                    SumH = 0.0;
                    for (int i = 0; i < NUMIN; i++)
                        SumH += tSet[p][i] * WeightIH[j][i];
                    Hidden[j] = 1.0 / (1.0 + exp(-SumH));
                }
                for (int k = 0; k < NUMOUT; k++)
                {
                    SumO = 0.0;
                    for (int j = 0; j < NUMHID; j++)
                        SumO += Hidden[j] * WeightHO[k][j];
                    Output[k] = 1.0 / (1.0 + exp(-SumO));
                    BError += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]);
                    DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]);
                }
                for (int j = 0; j < NUMHID; j++)
                {
                    SumDOW = 0.0;
                    for (int k = 0; k < NUMOUT; k++)
                        SumDOW += WeightHO[k][j] * DeltaO[k];
                    DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);
                    for (int i = 0; i < NUMIN; i++)
                        DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
                }
                for (int k = 0; k < NUMOUT; k++)
                    for (int j = 0; j < NUMHID; j++)
                        DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
            }
            Error += BError;
            for (int j = 0; j < NUMHID; j++)
                for (int i = 0; i < NUMIN; i++)
                {
                    WeightIH[j][i] += DeltaWeightIH[j][i];
                    if (j < NUMOUT && i < NUMHID)
                        WeightHO[j][i] += DeltaWeightHO[j][i];
                }

            // for (int k = 0; k < NUMOUT; k++)
            //     for (int j = 0; j < NUMHID; j++)
            //         WeightHO[k][j] += DeltaWeightHO[k][j];
        }

        MPI_Allreduce(MPI_IN_PLACE, WeightIH, NUMHID * NUMIN, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, DeltaWeightIH, NUMHID * NUMIN, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, WeightHO, NUMOUT * NUMHID, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, DeltaWeightHO, NUMOUT * NUMHID, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Faire tout dans une seule boucle plus rapide !
        for (int j = 0; j < NUMHID; j++)
            for (int i = 0; i < NUMIN; i++)
            {
                if (j < NUMOUT && i < NUMHID)
                {
                    WeightHO[j][i] /= numberOfProcess;
                    DeltaWeightHO[j][i] /= numberOfProcess;
                }
                WeightIH[j][i] /= numberOfProcess;
                DeltaWeightIH[j][i] /= numberOfProcess;
            }
        
        MPI_Allreduce(MPI_IN_PLACE, &Error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        Error = Error / ((NUMPAT / BSIZE) * BSIZE);

        if (!(epoch % 100) && rank == MASTER)
            printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
        if (Error < 0.0004 * ERROR_MULTIPLICATION)
        {
            if (rank == MASTER)
                printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);
            break;
        }
    }

    freeTSet(NUMPAT, tSet);

    if (rank == MASTER)
        printf("END TRAINING\n");
}

void printRecognized(int p, float Output[])
{
    int imax = 0;

    for (int i = 1; i < NUMOUT; i++)
        if (Output[i] > Output[imax])
            imax = i;
    if (DISPLAY_OUTPUT != 0)
        printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p]);
    if (imax == Validation[p])
        total++;
    if (DISPLAY_OUTPUT != 0)
    {
        for (int k = 0; k < NUMOUT; k++)
            printf("\t%f\t", Output[k]);
        printf("\n");
    }
}

void runN()
{
    char **rSet;
    char *fname[NUMRPAT];

    if ((rSet = loadPatternSet(NUMRPAT, "optdigits.cv", 0)) == NULL)
    {
        printf("Error!!\n");
        exit(-1);
    }

    float Hidden[NUMHID], Output[NUMOUT];

    for (int p = 0; p < NUMRPAT; p++)
    { // repeat for all the recognition patterns
        for (int j = 0; j < NUMHID; j++)
        { // compute hidden unit activations
            float SumH = 0.0;
            for (int i = 0; i < NUMIN; i++)
                SumH += rSet[p][i] * WeightIH[j][i];
            Hidden[j] = 1.0 / (1.0 + exp(-SumH));
        }

        for (int k = 0; k < NUMOUT; k++)
        { // compute output unit activations
            float SumO = 0.0;
            for (int j = 0; j < NUMHID; j++)
                SumO += Hidden[j] * WeightHO[k][j];
            Output[k] = 1.0 / (1.0 + exp(-SumO)); // Sigmoidal Outputs
        }
        printRecognized(p, Output);
    }

    printf("\nTotal encerts = %d\n", total);

    freeTSet(NUMRPAT, rSet);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcess);

    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    trainN();
    if (rank == MASTER)
        runN();

    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    if (rank == MASTER)
        printf("\n\nGoodbye! (%.3f sec)\n\n", elapsed);
    MPI_Finalize();
    return 0;
}

/*******************************************************************************/
