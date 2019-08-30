#include <mpi.h>
#include <math.h>

#include "manager.h"
#include "task.h"
#include "common.h"

void split(task_t* task, int rank, int target) //std::vector<task_t> tasks)
{
    task_t new_task;
    int size = task->size;
    int new_size = floor(task->size / 2);

    task->size = new_size;

    // 'Left split' (original task receives the left side). Might be more favorable to do a right split in some situations.
    new_task.size = size - new_size;
    new_task.A = &task->A[new_size - 1];
    new_task.C = &task->C[new_size];
    new_task.offset = task->offset + new_size;
    new_task.barrier = task->barrier;
    new_task.start_barrier = task->start_barrier;
    new_task.prev.rank = rank;
    new_task.prev.id = task->id;
    new_task.next.rank = task->next.rank;
    new_task.next.id = task->next.id;

    // Find devices.
    // if(freeDeviceOnNode()) {
        //task->next_rank = rank;
        // type = GPU / CPU
        // if(type == GPU) { allocCuda(task); }
        // tasks.push_back(new_task); // Copies value, thus should be done ASAP, and then tasks[i].A = ...

        // MPI_Send(MANAGER, UPDATE, ...)
        // MPI_Send(MANAGER, REGISTER, ...);
    // } else {
        // int target = findSomeFreeNode();
        task->next.rank = target;

        printf("(%d) Initiating task send to %d\n", rank, target);

        MPI_Status status;
        MPI_Send(&new_task.size, 1, MPI_INT, target, SPLIT, *task->manager);
        MPI_Send(new_task.A, new_task.size + 2, MPI_FLOAT, target, SPLIT, *task->manager);
        MPI_Send(new_task.C, new_task.size, MPI_FLOAT, target, SPLIT, *task->manager);
        MPI_Send(&new_task.next.rank, 1, MPI_INT, target, SPLIT, *task->manager);
        MPI_Send(&new_task.next.id, 1, MPI_INT, target, SPLIT, *task->manager);
        MPI_Send(&new_task.prev.rank, 1, MPI_INT, target, SPLIT, *task->manager);
        MPI_Send(&new_task.prev.id, 1, MPI_INT, target, SPLIT, *task->manager);
        MPI_Send(&new_task.offset, 1, MPI_INT, target, SPLIT, *task->manager);

        MPI_Recv(&task->next.id, 1, MPI_INT, target, SPLIT, *task->manager, MPI_STATUS_IGNORE);

        MPI_Send(&task->id, 1, MPI_INT, MANAGER_RANK, UPDATE, *task->manager);
        MPI_Send(&task->size, 1, MPI_INT, MANAGER_RANK, UPDATE, *task->manager);

        printf("(%d) Completed task send to %d\n", rank, target);
   //}
}

void receive_split(int rank, int source, std::vector<task_t> &tasks, MPI_Comm& manager)
{
    printf("(%d) Initiating task receive from %d\n", rank, source);

    tasks.emplace_back();
    int i = tasks.size() - 1;

    MPI_Status status;
    MPI_Recv(&tasks[i].size, 1, MPI_INT, source, SPLIT, manager, &status);
    tasks[i].A = (float*) malloc((tasks[i].size + 2) * sizeof(float));
    tasks[i].C = (float*) malloc((tasks[i].size) * sizeof(float));

    MPI_Recv(tasks[i].A, tasks[i].size + 2, MPI_FLOAT, source, SPLIT, manager, &status);
    MPI_Recv(tasks[i].C, tasks[i].size, MPI_FLOAT, source, SPLIT, manager, &status);
    MPI_Recv(&tasks[i].next.rank, 1, MPI_INT, source, SPLIT, manager, &status);
    MPI_Recv(&tasks[i].next.id, 1, MPI_INT, source, SPLIT, manager, &status);
    MPI_Recv(&tasks[i].prev.rank, 1, MPI_INT, source, SPLIT, manager, &status);
    MPI_Recv(&tasks[i].prev.id, 1, MPI_INT, source, SPLIT, manager, &status);
    MPI_Recv(&tasks[i].offset, 1, MPI_INT, source, SPLIT, manager, MPI_STATUS_IGNORE);
    tasks[i].id = id++;

    tasks[i].A = &tasks[i].A[1];

    printf("(%d) Received task of size: %d\n", rank, tasks[i].size);
    printf("R-Task: size: %d\n", tasks[i].size);
    printf("R-Task: next: %d:%d\n", tasks[i].next.rank, tasks[i].next.id);
    printf("R-Task: prev: %d:%d\n", tasks[i].prev.rank, tasks[i].prev.rank);
    printf("R-Task: A: %p\n", tasks[i].A);
    printf("R-Task: C: %p\n", tasks[i].C);
    for(int j = -1; j < tasks[i].size + 1; j++) {
        printf("R: A[%d]: %f\n", j, tasks[i].A[j]);
    }
    for(int j = 0; j < tasks[i].size; j++) {
        printf("R: C[%d]: %f\n", j, tasks[i].C[j]);
    }

    printf("R-Task: offset: %d\n", tasks[i].offset);

    MPI_Send(&tasks[i].id, 1, MPI_INT, source, SPLIT, manager);

    MPI_Send(&tasks[i].offset, 1, MPI_INT, MANAGER_RANK, REGISTER, manager);
    MPI_Send(&tasks[i].id, 1, MPI_INT, MANAGER_RANK, REGISTER, manager);
    MPI_Send(&tasks[i].size, 1, MPI_INT, MANAGER_RANK, REGISTER, manager);
}

void fetch_and_update_neighbours(int rank, task_t* task, std::vector<MPI_Receive_req> &requests, std::vector<int> &types, bool will_split)
{
    ref_t prevref = task->prev;
    ref_t nextref = task->next;

    if(prevref.rank != -1) {
        MPI_Request send_request;
        MPI_Request request;

        MPI_Isend(&task->A[0], 1, MPI_FLOAT, prevref.rank, construct_tag(prevref.id, 1, will_split ? WILL_SPLIT : DEFAULT), MPI_COMM_WORLD, &send_request);
        // MPI_Irecv(&task->A[-1], 1, MPI_FLOAT, prevref.rank, task->id * 100 + 0, MPI_COMM_WORLD, &request);
        requests.emplace_back();
        int i = requests.size() - 1;
        requests[i].buffer = &task->A[-1];
        requests[i].count = 1;
        requests[i].datatype = MPI_FLOAT;
        requests[i].source = prevref.rank;
        requests[i].tag_matcher = [task](int tag) { return match_tag(task->id, 0, -1, tag); };

        printf("(%d:%d) sending and receiving @ prev %d:%d. Split: %s\n", rank, task->id, prevref.rank, prevref.id, will_split ? "true" : "false");
        types.push_back(PREV_TYPE);
    }

    if(nextref.rank != -1) {
        MPI_Request send_request;
        MPI_Request request;

        MPI_Isend(&task->A[task->size - 1], 1, MPI_FLOAT, nextref.rank, construct_tag(nextref.id, 0, will_split ? WILL_SPLIT : DEFAULT), MPI_COMM_WORLD, &send_request);
        // MPI_Irecv(&task->A[task->size], 1, MPI_FLOAT, nextref.rank, task->id * 100 + 1, MPI_COMM_WORLD, &request);
        requests.emplace_back();
        int i = requests.size() - 1;
        requests[i].buffer = &task->A[task->size];
        requests[i].count = 1;
        requests[i].datatype = MPI_FLOAT;
        requests[i].source = nextref.rank;
        requests[i].tag_matcher = [task](int tag) { return match_tag(task->id, 1, -1, tag); };

        printf("(%d:%d) sending and receiving @ next %d:%d. Split: %s\n", rank, task->id, nextref.rank, nextref.id, will_split ? "true" : "false");
        types.push_back(NEXT_TYPE);
    }
}

void init_tasks(std::vector<task_t> &tasks, int task_count, Barrier* barrier, Barrier* start_barrier, MPI_Comm* manager, int active_devices)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int length = ceil(N / active_devices);
    int start = rank * length;

    if(rank == active_devices - 1) {
        length = N - length * (active_devices - 1);
    }

    printf("(%d) Length: %d\n", rank, length);

    // float* A = (float*) malloc(sizeof(float) * (length + 2));
    // float* C = (float*) malloc(sizeof(float) * length);
    // for (int i = 0; i < length + 2; i++)
    // {
    //     A[i] = 1;
    // }

    int sizePerDevice = ceil(length / task_count);
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
        tasks[i].start_barrier = start_barrier;
        tasks[i].manager = manager;
        tasks[i].start_iteration = 0;

        tasks[i].A = (float*) malloc(sizeof(float) * (tasks[i].size + 2));
        for(int j = 0; j < tasks[i].size + 2; j++) {
            tasks[i].A[j] = 1;
        }

        tasks[i].A = &tasks[i].A[1];

        tasks[i].C = (float*) malloc(sizeof(float) * tasks[i].size);
    }

    // Allocate GPU memory for the CUDA tasks
    int id = 0;
    for(int i = 0; i < tasks.size(); i++) {
        if(tasks[i].type == GPU) {
            tasks[i].cuda.id = id;
            id++;
            alloc_cuda(&tasks[i]);
        }
    }
}
