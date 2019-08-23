#pragma once

#include <vector>
#include <mpi.h>

#include "barrier.h"

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
    float* location;
    bool contiguous;
};

struct task_t {
    devicetype type;

    int size;

    float* A;
    float* C;
    int start_iteration;

    cudamem_t cuda;
    Barrier* barrier;

    ref_t next;
    ref_t prev;
};

void split(task_t* task, int rank, std::vector<task_t> &tasks);
void receive_split(int rank, int source, std::vector<task_t> &tasks);

void fetch_and_update_neighbours(int rank, task_t* task, std::vector<MPI_Request> &requests);

void init_tasks(std::vector<task_t> &tasks, int task_count, Barrier* barrier, int active_devices);