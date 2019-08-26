#include <mpi.h>
#include <math.h>

#include "manager.h"
#include "task.h"
#include "common.h"

void split(task_t* task, int rank, int target, std::vector<task_t> tasks)
{
    task_t new_task;
    int size = task->size;
    int new_size = floor(task->size / 2);

    task->size = new_size;

    // 'Left split' (original task receives the left side). Might be more favorable to do a right split in some situations.
    new_task.size = size - new_size;
    new_task.A = &task->A[new_size - 1];
    new_task.C = &task->C[new_size];
    new_task.barrier = task->barrier;
    new_task.prev.rank = rank;
    new_task.next.rank = task->next.rank;

    // Find devices.
    // if(freeDeviceOnNode()) {
        //task->next_rank = rank;
        // type = GPU / CPU
        // if(type == GPU) { allocCuda(task); }
        // tasks.push_back(new_task); // Copies value, thus should be done ASAP, and then tasks[i].A = ...
    // } else {
        // int target = findSomeFreeNode();
        task->next.rank = target;

        printf("(%d) Initiating task send to %d\n", rank, target);

        MPI_Status status;
        MPI_Send(&new_task.size, 1, MPI_INT, target, SPLIT, MPI_COMM_WORLD);
        MPI_Send(new_task.A, new_task.size + 2, MPI_FLOAT, target, SPLIT, MPI_COMM_WORLD);
        MPI_Send(new_task.C, new_task.size, MPI_FLOAT, target, SPLIT, MPI_COMM_WORLD);
        MPI_Send(&new_task.next.rank, 1, MPI_INT, target, SPLIT, MPI_COMM_WORLD);
        MPI_Send(&new_task.prev.rank, 1, MPI_INT, target, SPLIT, MPI_COMM_WORLD);
        MPI_Send(&new_task.offset, 1, MPI_INT, target, SPLIT, MPI_COMM_WORLD);
   //}
}

void receive_split(int rank, int source, std::vector<task_t> &tasks)
{
    printf("(%d) Initiating task receive from %d\n", rank, source);

    tasks.emplace_back();
    int i = tasks.size() - 1;

    MPI_Status status;
    MPI_Recv(&tasks[i].size, 1, MPI_INT, source, SPLIT, MPI_COMM_WORLD, &status);
    tasks[i].A = (float*) malloc((tasks[i].size + 2) * sizeof(float));
    tasks[i].C = (float*) malloc((tasks[i].size) * sizeof(float));

    MPI_Recv(tasks[i].A, tasks[i].size + 2, MPI_FLOAT, source, SPLIT, MPI_COMM_WORLD, &status);
    MPI_Recv(tasks[i].C, tasks[i].size, MPI_FLOAT, source, SPLIT, MPI_COMM_WORLD, &status);
    MPI_Recv(&tasks[i].next.rank, 1, MPI_INT, source, SPLIT, MPI_COMM_WORLD, &status);
    MPI_Recv(&tasks[i].prev.rank, 1, MPI_INT, source, SPLIT, MPI_COMM_WORLD, &status);
    MPI_Recv(&tasks[i].offset, 1, MPI_INT, source, SPLIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("(%d) Received task of size: %d\n", rank, tasks[i].size);
    printf("R-Task: size: %d\n", tasks[i].size);
    printf("R-Task: next: %d\n", tasks[i].next.rank);
    printf("R-Task: prev: %d\n", tasks[i].prev.rank);
    printf("R-Task: A: %p\n", tasks[i].A);
    printf("R-Task: C: %p\n", tasks[i].C);
    printf("R-Task: offset: %p\n", tasks[i].offset);

}

void fetch_and_update_neighbours(int rank, task_t* task, std::vector<MPI_Request> &requests, std::vector<int> &types, bool will_split)
{
    ref_t prevref = task->prev;
    ref_t nextref = task->next;

    if(prevref.rank != -1) {
        MPI_Request send_request;
        MPI_Request request;

        printf("Sending : %f\n", task->A[0]);
        printf("receiving over %f\n", task->A[-1]);

        MPI_Isend(&task->A[0], 1, MPI_FLOAT, prevref.rank, prevref.id, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(&task->A[-1], 1, MPI_FLOAT, prevref.rank, task->id, MPI_COMM_WORLD, &request);

        printf("(%d:%d) waiting for prev %d:%d\n", rank, task->id, prevref.rank, prevref.id);
        requests.push_back(request);
        requests.push_back(send_request);
        types.push_back(PREV_TYPE);
    }

    if(nextref.rank != -1) {
        MPI_Request send_request;
        MPI_Request request;

        printf("NSending: %f\n", task->A[task->size - 1]);
        printf("NReceiving over %f (%p) to %d:%d\n", task->A[task->size], &task->A[task->size], nextref.rank, task->id);


        MPI_Isend(&task->A[task->size - 1], 1, MPI_FLOAT, nextref.rank, nextref.id, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(&task->A[task->size], 1, MPI_FLOAT, nextref.rank, task->id, MPI_COMM_WORLD, &request);

        printf("(%d:%d) waiting for next %d:%d\n", rank, task->id, nextref.rank, nextref.id);
        requests.push_back(request);
        requests.push_back(send_request);
        types.push_back(NEXT_TYPE);
    }
}

void init_tasks(std::vector<task_t> &tasks, int task_count, Barrier* barrier, int active_devices)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int length = ceil(N / active_devices);
    int start = rank * length;

    // float* A = (float*) malloc(sizeof(float) * (length + 2));
    // float* C = (float*) malloc(sizeof(float) * length);
    // for (int i = 0; i < length + 2; i++)
    // {
    //     A[i] = 1;
    // }

    int sizePerDevice = ceil(length / (task_count));
    for(int i = 0; i < task_count; i++)
    {
        tasks.emplace_back();
        tasks[i].type = i == 0 ? CPU : GPU;
        tasks[i].id = id++;
        int offset = sizePerDevice * i;
        tasks[i].offset = offset + start;
        // tasks[i].A = &A[offset + 1];
        // tasks[i].C = &C[offset];

        if(i == task_count - 1) {
            tasks[i].size = length - sizePerDevice * (task_count - 1);
        } else {
            tasks[i].size = sizePerDevice;
        }

        tasks[i].barrier = barrier;
        tasks[i].start_iteration = 0;

        tasks[i].A = (float*) malloc(sizeof(float) * (tasks[i].size + 2));
        for(int j = 0; j < tasks[i].size + 2; j++) {
            tasks[i].A[j] = 1;
        }

        tasks[i].A = &tasks[i].A[1];
        printf("Legal op: %f\n", tasks[i].A[-1]);

        tasks[i].C = (float*) malloc(sizeof(float) * tasks[i].size);
        printf("task: %p\n", &tasks[i]);
    }

    // Point to 'next' and 'prev' patch locations.
    // for(int i = 0; i < task_count; i++) {
    //     if(i == 0) {
    //         if(rank == 0) {
    //             tasks[i].prev.rank = -1;
    //         } else {
    //             tasks[i].prev.rank = rank - 1;
    //         }
    //     } else {
    //         tasks[i].prev.rank = rank;
    //     }

    //     if(i == task_count - 1) {
    //         if(rank == active_devices - 1) {
    //             tasks[i].next.rank = -1;
    //         } else {
    //             tasks[i].next.rank = rank + 1;
    //         }
    //     } else {
    //         tasks[i].next.rank = rank;
    //     }
    // }

    // Allocate GPU memory for the CUDA tasks
    int id = 0;
    for(int i = 0; i < tasks.size(); i++) {
        if(tasks[i].type == GPU) {
            printf("Initing cuda task: %p, id: %d\n", &tasks[i], id);
            tasks[i].cuda.id = id;
            id++;
            alloc_cuda(&tasks[i]);
        }
    }
}
