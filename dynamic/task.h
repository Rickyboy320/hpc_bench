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
    int offset;
    float* A;
    float* C;
    bool done;
    cudamem_t cuda;
    Barrier* barrier;
};

task_t split(task_t* task);
void init_tasks(task_t* tasks, int task_count, Barrier* barrier, int active_devices);
