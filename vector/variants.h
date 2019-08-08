#pragma once

#include "common.h"

enum Variant { pthreads, cthreads, openmp, cuda };

void run_pthread_variant(int rank, int gpu_count, task_t tasks[]);
void run_cthread_variant(int rank, int gpu_count, task_t tasks[]);
void run_openmp_variant(int rank, int gpu_count, task_t tasks[]);
void run_cuda_variant(int rank, int gpu_count, task_t tasks[]);
