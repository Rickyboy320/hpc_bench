#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "barrier.h"

#define RUNS 10
#define CYCLES 10

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
    float* C = task->C;
    int size = task->size;
    int offset = task->offset;

    while(!task->done) {
        // Run task (sum neighbours) with OpenMP
        int i;
        #pragma omp parallel for private(i) shared(A,C)
        for (i = offset; i < offset + size; i++)
        {
            int prev = i == 0 ? N - 1 : i - 1;
            int next = i == N - 1 ? 0 : i + 1;

            C[i] = A[prev] + A[i] + A[next];
        }

        printf("Waiting barrier OpenMP\n");
        task->barrier->wait();
    }

    printf("omp.done()!\n");
    pthread_exit(NULL);
}

void run_cthread_variant(int rank, int gpu_count, task_t tasks[], Barrier* barrier)
{
    std::thread threads[gpu_count + 1];

    threads[0] = std::thread(run_openmp, &tasks[0]);

    for(int i = 1; i <= gpu_count; i++)
    {
        threads[i] = std::thread(run_cuda, &tasks[i]);
    }

    for(int i = 0; i < CYCLES; i++) {
        if(i == CYCLES - 1) {
            for(int i = 0; i < gpu_count + 1; i++) {
                tasks[i].done = true;
                printf("Setting done for %d\n", i);
            }
        }

        printf("Waiting barrier main\n");
        barrier->wait();
        barrier->reset();

        printf("Waiting MPI\n");

        //  Sync to wait on all processes.
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Wait for all tasks to complete.
    for(int i = 0; i <= gpu_count; i++)
    {
        printf("Joining %d\n", i);
        threads[i].join();
    }
}

int main(int argc, char** argv)
{
    // Initialize vectors
    float A[N];
    float C[N];

    int rank = init_mpi();
    init_openmp();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int length = ceil(N / world_size);
    int start = rank * length;

    for (int i = start; i < start + length; i++)
    {
        A[i] = rank + 1;
    }

    // Distribute tasks evenly over CPU (OpenMP) and available GPUs (CUDA)
    int gpu_count = init_cuda();

    printf("Rank: %d: Count GPU devices: %d\n", rank, gpu_count);

    Barrier barrier(gpu_count + 2);

    task_t tasks[gpu_count + 1];

    int sizePerDevice = ceil(length / (gpu_count + 1.0));
    for(int i = 0; i < gpu_count + 1; i++)
    {
        tasks[i].cuda.id = i;
        tasks[i].A = &A[start + sizePerDevice * i];
        tasks[i].C = &C[start + sizePerDevice * i];

        if(i == gpu_count) {
            tasks[i].size = N - sizePerDevice * gpu_count;
        } else {
            tasks[i].size = sizePerDevice;
        }

        tasks[i].barrier = &barrier;
        tasks[i].done = false;
    }

    // Allocate GPU memory for the CUDA tasks
    for(int i = 1; i < gpu_count + 1; i++)
    {
        alloc_cuda(&tasks[i]);
    }

    //  Sync for 'equal' starts.
    MPI_Barrier(MPI_COMM_WORLD);

    // Start benchmark                 ========================================
    for(int i = 0; i < RUNS; i++)
    {
        for(int i = 0; i <= gpu_count; i++) {
            tasks[i].done = false;
        }

        printf("Running: %d\n", i);
        // Run tasks
        run_cthread_variant(rank, gpu_count, tasks, &barrier);
    }
    // End benchmark                   ========================================

    for(int i = 1; i < gpu_count + 1; i++)
    {
        dealloc_cuda(&tasks[i]);
    }

    // Communicate result over MPI & verify.
     if(rank == 0)
    {
        // MPI_Request request;
        // MPI_Irecv(&C[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &request);
        // MPI_Send(&C[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);

        // MPI_Status status;
        // MPI_Wait(&request, &status);


        for (int i = 0; i < N; i++)
        {
            int prev = i == 0 ? N - 1 : i - 1;
            int next = i == N - 1 ? 0 : i + 1;

            if(fabs(A[prev] + A[i] + A[next] - C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }
    }

    MPI_Finalize();
}