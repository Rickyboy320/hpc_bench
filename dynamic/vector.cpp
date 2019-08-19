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

    printf("ompt task: %p\n", task);

    printf("omp A: %p\n", A);
    printf("omp done: %s\n", task->done ? "true" : "false");

    while(!task->done) {
        // Run task (sum neighbours) with OpenMP
        int i;
        #pragma omp parallel for private(i) shared(A,C)
        for (i = 0; i < size; i++)
        {
            C[i] = A[i] + A[i-1] + A[i+1];
            printf("A: %f A-1 %f A+1 %f C: %f\n", A[i], A[i-1], A[i+1], C[i]);
        }

        printf("omp barrier\n");
        task->barrier->wait();
        task->barrier->wait();
    }

    printf("omp done\n");
    pthread_exit(NULL);
}

void run_cthread_variant(int rank, int task_count, std::vector<task_t> &tasks, Barrier* barrier)
{
    std::vector<std::thread> threads;

    // Pre communication:
    std::vector<MPI_Request> requests;
    for(int i = 0; i < tasks.size(); i++) {
        fetch_and_update_neighbours(rank, &tasks[i], requests);
    }

    printf("(%d) Pre Waiting all\n", rank);
    if(!requests.empty()) {
        MPI_Status* statuses;
        MPI_Waitall(requests.size(), &requests[0], statuses);
    }

    printf("(%d) Starting threads: %d\n", rank, tasks.size());
    for(int i = 0; i < tasks.size(); i++) {
        if(tasks[i].type == CPU) {
            threads.push_back(std::thread(run_openmp, &tasks[i]));
        } else if(tasks[i].type == GPU) {
            threads.push_back(std::thread(run_cuda, &tasks[i]));
        } else {
            printf("WARNING: task without type: %p\n", &tasks[i]);
            throw std::exception();
        }
    }

    for(int c = 0; c < CYCLES; c++) {
        printf("(%d) Waiting barrier main\n", rank);
        barrier->wait();

        if(c == CYCLES - 1) {
            for(int i = 0; i < tasks.size(); i++) {
                tasks[i].done = true;
            }
        }

        // Switch buffers (slow quick & dirty variant)
        for(int i = 0; i < task_count; i++) {
            for(int j = 0; j < tasks[i].size; j++) {
                printf("C%d: (%d) [%d] %d: %f\n", c, rank, i, j, tasks[i].C[j]);

                tasks[i].A[j] = tasks[i].C[j];
            }
        }

        printf("(%d) Updating neighbours\n", rank);
        std::vector<MPI_Request> requests;
        for(int i = 0; i < tasks.size(); i++) {
            fetch_and_update_neighbours(rank, &tasks[i], requests);

            // TODO: now this deadlocks because 3rd patch does not know that the patch changed.
        }

        MPI_Status* statuses;
        MPI_Waitall(requests.size(), &requests[0], statuses);

        // Split
        if(c == 3 && rank == 0) {
            // Arbitrarily (as a test) decide to split.
            split(&tasks[0], rank, tasks);
        }

        if(c == 3 && rank == 1) {
            receive_split(rank, 0, tasks);
        }

        printf("(%d) Waiting MPI\n", rank);

        //  Sync to wait on all processes.
        MPI_Barrier(MPI_COMM_WORLD);
        barrier->wait();
    }

    // Wait for all tasks to complete.
    for(int i = 0; i < task_count; i++)
    {
        printf("(%d) Joining %d\n", rank, i);
        threads[i].join();
    }
}

int main(int argc, char** argv)
{
    // Parse command line args
    int active_devices = 1;
    if(argc > 1) {
        active_devices = std::stoi(argv[1]);
    }

    int rank = init();

    // Distribute tasks evenly over nodes and devices.
    int gpu_count = init_cuda();
    int task_count = rank < active_devices ? gpu_count + 1 : 0;

    printf("Rank: %d: Count GPU devices: %d. Tasks: %d\n", rank, gpu_count, task_count);

    Barrier barrier(task_count + 1);

    std::vector<task_t> tasks;
    if(task_count > 0) {
        init_tasks(tasks, task_count, &barrier, active_devices);
    }

    //  Sync for 'equal' starts.
    MPI_Barrier(MPI_COMM_WORLD);

    // Run tasks
    run_cthread_variant(rank, task_count, tasks, &barrier);


    for(int i = 0; i < task_count; i++)
    {
        if(tasks[i].type == GPU) {
            dealloc_cuda(&tasks[i]);
        }
    }

    // Communicate result over MPI & verify.
    for(int i = 0; i < tasks.size(); i++) {
        for(int j = 0; j < tasks[i].size; j++) {
            printf("(%d) [%d] %d: %f\n", rank, i, j, tasks[i].C[j]);
        }
    }

    if(rank == 0)
    {
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