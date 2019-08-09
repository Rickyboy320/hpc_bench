#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

struct cudamem_t {
    int size;
    float* A;
    float* B;
    float* C;
};

struct task_t {
    int id;
    int size;
    float* A;
    float* B;
    float* C;
    bool is_threads = false;
    cudamem_t cudamem;

};

int init_cuda();
void alloc_cuda(task_t* task);
void dealloc_cuda(task_t* task);
void* run_cuda(void* v_task);
cudaStream_t* run_cuda_streams(int gpu_count, task_t tasks[]);
void sync_cuda_streams(int gpu_count, cudaStream_t streams[]);
void* run_openmp(void* v_task);
