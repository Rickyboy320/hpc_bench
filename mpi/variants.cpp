#include <thread>
#include <future>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>

#include "common.h"

int MPI_WAKE_UP = 0;

void run_openmp(task_t task)
{
    float* A = task.A;
    float* B = task.B;
    float* C = task.C;
    int size = task.size;

    // Run task (vector addition) with OpenMP
    int i;
    #pragma omp parallel for private(i) shared(A,B,C)
    for (i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void run_spawn_variant(int argc, char** argv, int rank, task_t task)
{
    MPI_Info info;
    MPI_Info_create(&info);

    MPI_Comm parentcomm, intercomm;
    MPI_Comm_get_parent(&parentcomm);

    if (parentcomm == MPI_COMM_NULL) {
        // Work
        run_openmp(task);

        // Spawn child processes
        int sub_processes = 8;
        printf("Rank %d spawning %d process(es)\n", rank, sub_processes);

        int errcodes[sub_processes];
        if(argc > 1) {
            MPI_Comm_spawn(argv[0], &argv[1], sub_processes, info, 0, MPI_COMM_WORLD, &intercomm, errcodes);
        } else {
            MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, sub_processes, info, 0, MPI_COMM_WORLD, &intercomm, errcodes);
        }

        int size = 0;
        MPI_Comm_size(intercomm, &size);

        MPI_Barrier(intercomm);
    }
    else
    {
        int size = 0;
        MPI_Comm_size(parentcomm, &size);

        MPI_Barrier(parentcomm);
    }
}

void run_idle_variant(int rank, int world_size, task_t task)
{
    if(rank != 0) {
        // Idle
        int buf[1];
        MPI_Status status;
        MPI_Recv(&buf, 1, MPI_INT, 0, MPI_WAKE_UP, MPI_COMM_WORLD, &status);
    } else {
        // Work
        run_openmp(task);

        // Wake up.
        int buf[world_size - 1];
        MPI_Request requests[world_size - 1];
        for(int i = 1; i < world_size; i++) {
            buf[i - 1] = i;
            MPI_Isend(&buf[i - 1], 1, MPI_INT, i, MPI_WAKE_UP, MPI_COMM_WORLD, &requests[i - 1]);
        }
    }
}

void run_single_variant(task_t task)
{
    // Work
    run_openmp(task);
}