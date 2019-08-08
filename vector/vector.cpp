#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "variants.h"
#include "common.h"

#define N 1000

void init_openmp()
{
    omp_set_num_threads(omp_get_num_procs());
    printf("Number omp procs: %d\n", omp_get_num_procs());
}

int init_mpi()
{
    MPI_Init(NULL, NULL);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Node %s = rank %d\n", processor_name, world_rank);

    return world_rank;
}

void* run_openmp(void* v_task)
{
    // Unpack task
    task_t* task = (task_t*) v_task;
    float* A = task->A;
    float* B = task->B;
    float* C = task->C;
    int size = task->size;

    // Run task (vector addition) with OpenMP
    int i;
    #pragma omp parallel for private(i) shared(A,B,C)
    for (i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }

    if(task->is_threads)
    {
        pthread_exit(NULL);
    }
    else
    {
        return NULL;
    }
}

int main(int argc, char** argv)
{
    Variant variant = openmp;
    for(int i = 1; i < argc; i++)
    {
        char* arg = argv[i];
        if(strcmp(arg, "--openmp") == 0) {
            variant = openmp;
        } else if(strcmp(arg, "--pthreads") == 0) {
            variant = pthreads;
        } else if(strcmp(arg, "--cthreads") == 0) {
            variant = cthreads;
        } else if(strcmp(arg, "--cuda") == 0) {
            variant = cuda;
        }
    }

    // Initialize vectors
    float A[N*2];
    float B[N*2];
    float C[N*2];

    int i;
    struct timeval tv1, tv2;
    struct timezone tz;
	double elapsed;

    int rank = init_mpi();
    init_openmp();

    // Init array

    int start = rank == 0 ? 0 : N;
    int receive = rank == 0 ? N : 0;

    for (i = start; i < start + N; i++)
    {
        A[i] = rank + 1;
        B[i] = rank + 1;
    }

    // Synchronize halfs of array with other node.
    MPI_Request requests[2];
    MPI_Irecv(&A[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&B[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &requests[1]);

    MPI_Send(&A[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
    MPI_Send(&B[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);

    MPI_Status statusses[2];
    MPI_Waitall(2, requests, statusses);


    // Distribute tasks evenly over CPU (OpenMP) and GPUs (CUDA)
    int gpu_count = init_cuda();

    printf("Rank: %d: Count GPU devices: %d\n", rank, gpu_count);

    task_t tasks[gpu_count + 1];

    int sizePerDevice = ceil(N / (gpu_count + 1.0));
    for(int i = 0; i < gpu_count + 1; i++)
    {
        tasks[i].A = &A[start + sizePerDevice * i];
        tasks[i].B = &B[start + sizePerDevice * i];
        tasks[i].C = &C[start + sizePerDevice * i];

        tasks[i].size = sizePerDevice;
    }

    //  Sync for 'equal' starts.
    MPI_Barrier(MPI_COMM_WORLD);


    // Start benchmark
    gettimeofday(&tv1, &tz);

    // Run tasks
    switch(variant)
    {
        case pthreads:
            printf("Selected variant: pthreads\n");
            run_pthread_variant(rank, gpu_count, tasks);
            break;
        case cthreads:
            printf("Selected variant: c++ threads\n");
            run_cthread_variant(rank, gpu_count, tasks);
            break;
        case openmp:
            printf("Selected variant: OpenMP threads\n");
            run_openmp_variant(rank, gpu_count, tasks);
            break;
        case cuda:
            printf("Selected variant: CUDA streams\n");
            run_cuda_variant(rank, gpu_count, tasks);
            break;
    }

    //  Sync to wait on all processes.
    MPI_Barrier(MPI_COMM_WORLD);


    gettimeofday(&tv2, &tz);
    // End benchmark


    MPI_Request request;
    MPI_Irecv(&C[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &request);
    MPI_Send(&C[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);

    MPI_Status status;
    MPI_Wait(&request, &status);

    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    if(rank == 0)
    {
        printf("Elapsed time = %f seconds.\n", elapsed);

        for (int i = 0; i < 2*N; ++i)
        {
            if (fabs(A[i] + B[i] - C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }
    }

    MPI_Finalize();
}