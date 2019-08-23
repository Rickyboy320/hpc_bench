#pragma once

#include <vector>
#include <mpi.h>

#include "barrier.h"

static const int PREV_TYPE = 0;
static const int NEXT_TYPE = 1;

enum devicetype {
    NONE,
    CPU,
    GPU
};

struct cudamem_t {
    int id;
    int size;
    float* A;
    float* C;
};

struct ref_t {
    int rank;
};

struct task_t {
    devicetype type;

    int offset;
    int size;
    int id;

    float* A;
    float* C;
    int start_iteration;

    cudamem_t cuda;
    Barrier* barrier;

    ref_t next;
    ref_t prev;
};

void split(task_t* task, int rank, int target);
void receive_split(int rank, int source, std::vector<task_t> &tasks);

void fetch_and_update_neighbours(int rank, task_t* task, std::vector<MPI_Request> &requests, std::vector<int> &types, bool will_split);

void init_tasks(std::vector<task_t> &tasks, int task_count, Barrier* barrier, int active_devices);