#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

struct task_t {
    int size;
    float* A;
    float* B;
    float* C;
    bool is_threads = false;
};

int init_cuda();
void* run_cuda(void* v_task);
cudaStream_t* run_cuda_streams(int gpu_count, task_t tasks[]);
void sync_cuda_streams(int gpu_count, cudaStream_t streams[]);
void* run_openmp(void* v_task);
