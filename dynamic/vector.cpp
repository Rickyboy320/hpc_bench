#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "task.h"
#include "common.h"
#include "barrier.h"

#define CYCLES 10

int init()
{
    omp_set_num_threads(omp_get_num_procs());
    printf("Number omp procs: %d\n", omp_get_num_procs());

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
        for (i = 0; i < size; i++)
        {
            C[i] = A[i];
            if(i + offset > 0) { C[i] += A[i-1]; }
            if(i + offset < N) { C[i] += A[i+1]; }
        }

        printf("Waiting barrier OpenMP\n");
        task->barrier->wait();
        task->barrier->wait();
    }

    printf("omp.done()!\n");
    pthread_exit(NULL);
}

void run_cthread_variant(int rank, int task_count, task_t tasks[], Barrier* barrier)
{
    std::thread threads[task_count];

    for(int i = 0; i < task_count; i++) {
        if(tasks[i].type == CPU) {
            threads[i] = std::thread(run_openmp, &tasks[i]);
        } else {
            threads[i] = std::thread(run_cuda, &tasks[i]);
        }
    }

    for(int i = 0; i < CYCLES; i++) {
        printf("Waiting barrier main\n");
        barrier->wait();

        if(i == CYCLES - 1) {
            for(int i = 0; i < task_count; i++) {
                tasks[i].done = true;
            }
        }

        // Exchange MPI info.


        printf("Waiting MPI\n");

        //  Sync to wait on all processes.
        MPI_Barrier(MPI_COMM_WORLD);
        barrier->wait();
    }

    // Wait for all tasks to complete.
    for(int i = 0; i < task_count; i++)
    {
        printf("Joining %d\n", i);
        threads[i].join();
    }
}

int main(int argc, char** argv)
{
    // Parse command line args
    int active_devices = 1;
    if(argc > 1) {
        active_devices = std::stoi(argv[0]);
    }

    int rank = init();

    // Distribute tasks evenly over nodes and devices.
    int gpu_count = init_cuda();
    int task_count = rank < active_devices ? gpu_count + 1 : 0;

    printf("Rank: %d: Count GPU devices: %d. Tasks: %d\n", rank, gpu_count, task_count);

    Barrier barrier(task_count + 1);

    task_t tasks[task_count];
    init_tasks(tasks, task_count, &barrier, active_devices);

    //  Sync for 'equal' starts.
    MPI_Barrier(MPI_COMM_WORLD);

    // Run tasks
    run_cthread_variant(rank, task_count, tasks, &barrier);


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


        // TODO: readd check

        // for (int i = 0; i < N; i++)
        // {
        //     int sum = A[i] + (i == 0 ? 0 : A[i - 1]) + (i == N - 1 ? 0 : A[i + 1]);
        //     if(fabs(sum - C[i]) > 1e-5)
        //     {
        //         fprintf(stderr, "Result verification failed at element %d! Was: %f, should be: %f\n", i, C[i], sum);
        //         exit(EXIT_FAILURE);
        //     }
        // }
    }

    MPI_Finalize();
}