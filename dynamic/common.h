#pragma once

#include <stdio.h>
#include "barrier.h"
#include "task.h"

#define N 20

int init_cuda();
void alloc_cuda(task_t* task);
void dealloc_cuda(task_t* task);

void* run_cuda(void* v_task);
void* run_openmp(void* v_task);
