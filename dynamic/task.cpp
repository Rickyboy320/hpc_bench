#include <mpi.h>
#include <math.h>

#include "task.h"
#include "common.h"

task_t split(task_t* task, int rank)
{
    task_t new_task;
    int size = task->size;
    int new_size = floor(task->size / 2);

    task->size = new_size;

    // 'Left split' (original task receives the left side). Might be more favorable to do a right split in some situations.
    new_task.size = size - new_size;
    new_task.A = &task->A[new_size];
    new_task.C = &task->C[new_size];
    new_task.barrier = task->barrier;
    new_task.prev_rank = rank;
    new_task.next_rank = task->next_rank;

    // Find devices.
    // if(freeDeviceOnNode()) {
        //task->next_rank = rank;
        // type = GPU / CPU
        // if(type == GPU) { allocCuda(task); }
        //
    // } else {
        // int target = findSomeFreeNode();
        int target = 1;
        task->next_rank = target;

        printf("(%d) Initiating task send to %d\n", rank, target);

        MPI_Status status;
        MPI_Send(&new_task.size, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
        MPI_Send(new_task.A, new_task.size + 2, MPI_FLOAT, target, 0, MPI_COMM_WORLD);
        MPI_Send(new_task.C, new_task.size, MPI_FLOAT, target, 0, MPI_COMM_WORLD);
        MPI_Send(&new_task.next_rank, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
        MPI_Send(&new_task.prev_rank, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
    //}

    return new_task;
}

task_t receive_split(int rank, int source)
{
    printf("(%d) Initiating task receive from %d\n", rank, source);

    task_t new_task;
    MPI_Status status;
    MPI_Recv(&new_task.size, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
    new_task.A = (float*) malloc((new_task.size + 2) * sizeof(float));
    new_task.C = (float*) malloc((new_task.size) * sizeof(float));

    MPI_Recv(new_task.A, new_task.size + 2, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(new_task.C, new_task.size, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&new_task.next_rank, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&new_task.prev_rank, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);

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
