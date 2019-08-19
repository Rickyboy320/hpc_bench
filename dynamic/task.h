#pragma once
#include "barrier.h"

enum devicetype {
    CPU,
    GPU
};

struct cudamem_t {
    int id;
    int size;
    float* A;
    float* C;
};

struct task_t {
    devicetype type;

    int size;

    float* A;
    float* C;
    bool done;

    cudamem_t cuda;
    Barrier* barrier;

    int next_rank;
    float* next_A;
    int prev_rank;
    float* prev_A;
};

task_t split(task_t* task, int rank);
task_t receive_split(int rank, int source);

void init_tasks(task_t* tasks, int task_count, Barrier* barrier, int active_devices);
