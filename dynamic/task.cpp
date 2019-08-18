#include <mpi.h>
#include <math.h>

#include "task.h"
#include "common.h"

task_t split(task_t* task)
{
    task_t new_task;
    int size = task->size;
    int new_size = floor(task->size / 2);

    task->size = new_size;

    new_task.size = size - new_size;
    new_task.A = &task->A[new_size];
    new_task.C = &task->C[new_size];
    new_task.barrier = task->barrier;
    new_task.offset = task->offset + new_size;

    return new_task;
}

void init_tasks(task_t* tasks, int task_count, Barrier* barrier, int active_devices)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int length = ceil(N / active_devices);
    int start = rank * length;

    float* A = (float*) malloc(sizeof(float) * length + 2);
    float* C = (float*) malloc(sizeof(float) * length);
    for (int i = 1; i < 1 + length; i++)
    {
        A[i] = rank + 1;
    }

    int sizePerDevice = ceil(length / (task_count));
    for(int i = 0; i < task_count; i++)
    {
        tasks[i].type = i == 0 ? CPU : GPU;
        tasks[i].offset = start + sizePerDevice * i;
        tasks[i].A = &A[tasks[i].offset];
        tasks[i].C = &C[tasks[i].offset];

        if(i == task_count - 1) {
            tasks[i].size = N - sizePerDevice * (task_count - 1);
        } else {
            tasks[i].size = sizePerDevice;
        }

        tasks[i].barrier = barrier;
        tasks[i].done = false;
    }

    // Allocate GPU memory for the CUDA tasks
    for(int i = 1; i < task_count; i++)
    {
        tasks[i].cuda.id = i - 1;

        alloc_cuda(&tasks[i]);
    }
}
