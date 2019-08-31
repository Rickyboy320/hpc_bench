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
    float* A;
    float* C;
};

struct ref_t {
    int rank;
    int id;
};

struct task_t {
    devicetype type;

    int size;
    int offset;
    int id;

    float* A;
    float* C;
    int start_iteration;

    cudamem_t cuda;
    Barrier* barrier;
    Barrier* start_barrier;
    MPI_Comm* manager;

    ref_t next;
    ref_t prev;
};

struct MPI_Receive_req {
    bool completed;
    void* buffer;
    int count;
    MPI_Datatype datatype;
    int source;
    std::function<bool(int)> tag_matcher;
};

void split(task_t* task, int rank);
void receive_split(int rank, int source, std::vector<task_t> &tasks, MPI_Comm& manager);

void fetch_and_update_neighbours(int rank, task_t* task, std::vector<MPI_Receive_req> &requests, std::vector<int> &types, bool will_split);

void init_tasks(std::vector<task_t> &tasks, int task_count, Barrier* barrier, Barrier* start_barrier, MPI_Comm* manager, int active_devices);