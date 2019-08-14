#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>

#include "variants.h"
#include "common.h"

#define N 10000000

int main(int argc, char** argv)
{
    Variant variant = mpi_single;
    for(int i = 1; i < argc; i++)
    {
        char* arg = argv[i];
        if(strcmp(arg, "--idle") == 0) {
            variant = mpi_idle;
        } else if(strcmp(arg, "--spawn") == 0) {
            variant = mpi_spawn;
        } else if(strcmp(arg, "--single") == 0) {
            variant = mpi_single;
        }
    }

    omp_set_num_threads(omp_get_num_procs());
    printf("Number omp procs: %d\n", omp_get_num_procs());


    task_t task;

    float A[N];
    float B[N];
    float C[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = 1;
        B[i] = 2;
    }

    task.A = A;
    task.B = B;
    task.C = C;
    task.size = N;

    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Comm parentcomm;
    MPI_Comm_get_parent(&parentcomm);

    printf("Node %s = rank %d\n", processor_name, rank);

    // Benchmark  =====================
	double elapsed = 0;
    struct timeval tv1, tv2;
    struct timezone tz;

    if(parentcomm == MPI_COMM_NULL) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) {
            // Main process
            gettimeofday(&tv1, &tz);
        }
    }

    // Variants
    switch(variant) {
        case mpi_idle:
            printf("Running idle variant\n");
            run_idle_variant(rank, world_size, task);
            break;
        case mpi_spawn:
            printf("Running spawn variant\n");
            run_spawn_variant(argc, argv, rank, task);
            break;
        case mpi_single:
            printf("Running single variant\n");
            run_single_variant(task);
            break;
    }

    // End benchmark ==============================

    if(rank == 0 && parentcomm == MPI_COMM_NULL) {
        // Main process
        gettimeofday(&tv2, &tz);
        double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
        printf("Elapsed time: %f seconds", elapsed);

        for(int i = 0; i < N; i++) {
            if (fabs(A[i] + B[i] - C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }
    }

    MPI_Finalize();
}
