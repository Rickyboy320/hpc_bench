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

    return new_task;
}

void init_tasks(task_t* tasks, int task_count, Barrier* barrier, int active_devices)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int length = ceil(N / active_devices);
    int start = rank * length;

    printf("(%d) Initing with length: %d. Start: %d\n", rank, length, start);

    float* A = (float*) malloc(sizeof(float) * length + 2);
    float* C = (float*) malloc(sizeof(float) * length);
    for (int i = 0; i < length + 2; i++)
    {
        A[i] = 1;
    }

    int sizePerDevice = ceil(length / (task_count));
    for(int i = 0; i < task_count; i++)
    {
        tasks[i].type = i == 0 ? CPU : GPU;
        int offset = sizePerDevice * i;
        tasks[i].A = &A[offset + 1];
        tasks[i].C = &C[offset];

        if(i == task_count - 1) {
            tasks[i].size = length - sizePerDevice * (task_count - 1);
        } else {
            tasks[i].size = sizePerDevice;
        }

        printf("(%d) Setting task[%d] to size: %d\n", rank, i, tasks[i].size);

        tasks[i].barrier = barrier;
        tasks[i].done = false;
    }

    // Point to 'next' and 'prev' patch locations.
    for(int i = 0; i < task_count; i++) {
        if(i == 0) {
            tasks[i].prev_rank = rank == 0 ? -1 : rank - 1;
        } else {
            tasks[i].prev_rank = rank;
        }

        if(i == task_count - 1) {
            tasks[i].next_rank = rank == active_devices - 1 ? -1 : rank + 1;
        } else {
            tasks[i].next_rank = rank;
        }
    }

    // Allocate GPU memory for the CUDA tasks
    for(int i = 1; i < task_count; i++)
    {
        tasks[i].cuda.id = i - 1;

        alloc_cuda(&tasks[i]);
    }
}
