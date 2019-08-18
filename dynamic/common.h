#pragma once

#include <stdio.h>
#include "barrier.h"

#define N 10

struct cudamem_t {
    int id;
    int size;
    float* A;
    float* C;
};

struct task_t {
    int size;
    int offset;
    float* A;
    float* C;
    bool done;
    cudamem_t cuda;
    Barrier* barrier;
};

int init_cuda();
void alloc_cuda(task_t* task);
void dealloc_cuda(task_t* task);

void* run_cuda(void* v_task);
void* run_openmp(void* v_task);
