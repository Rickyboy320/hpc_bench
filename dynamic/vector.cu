#include <pthread.h>

#include "common.h"

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}


__global__ void
vectorAdd(const float *A, float *C, int size, int inset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < size)
    {
        C[i] = A[i + inset] + A[i - 1 + inset] + A[i + 1 + inset];
        printf("C[%d]: %f, A[%d]: %f, A[%d-1]: %f, A[%d+1]: %f\n", i, C[i], i, A[i + inset], i, A[i-1 + inset], i, A[i+1 + inset]);
    }
}

// Allocate cuda memory and pin host memory (required for async stream).
void alloc_cuda(task_t* task)
{
    cudaSetDevice(task->cuda.id);

    task->cuda.size = task->size * sizeof(float);

    // Allocate the device vectors
    cudaCheck(cudaMalloc((void **)&task->cuda.A, task->cuda.size + 2 * sizeof(float))); // Plus 'imported' neighbours.
    cudaCheck(cudaMalloc((void **)&task->cuda.C, task->cuda.size));
}

// Deallocate cuda memory and unpin host memory.
void dealloc_cuda(task_t* task)
{
    cudaSetDevice(task->cuda.id);

    // Free device global memory
    cudaCheck(cudaFree(task->cuda.A));
    cudaCheck(cudaFree(task->cuda.C));
}

// Run the cuda task (on the 'thread stream').
void* run_cuda(void* v_task)
{
    task_t* task = (task_t*) v_task;

    printf("Setting device: %d\n", task->cuda.id);
    cudaSetDevice(task->cuda.id);

    while(!task->done) {
        printf("Cuda memcpy h2d: %d\n", task->cuda.id);
        // Copy the host input vectors A and B H2D.

        int inset = 0;
        cudaCheck(cudaMemcpy(task->cuda.A, &task->A[-1], task->cuda.size + 2 * sizeof(float), cudaMemcpyHostToDevice));
        inset = 1;

        // Launch the vector-add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (task->size + threadsPerBlock - 1) / threadsPerBlock;

        printf("Cuda kernel: %d\n", task->cuda.id);
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0>>>(task->cuda.A, task->cuda.C, task->size, inset);

        // Copy the device result vector D2H.
        printf("Cuda memcpy d2h: %d\n", task->cuda.id);
        cudaCheck(cudaMemcpy(task->C, task->cuda.C, task->cuda.size, cudaMemcpyDeviceToHost));

        printf("Cuda sync: %d\n", task->cuda.id);
        cudaCheck(cudaDeviceSynchronize());

        printf("Waiting barrier Cuda: %d\n", task->cuda.id);
        task->barrier->wait();
        task->barrier->wait();
    }

    printf("cud done: %d\n", task->cuda.id);
    pthread_exit(NULL);
}

// Get the number of available GPUs.
int init_cuda()
{
    int gpu_count;
    cudaError_t cerr = cudaGetDeviceCount(&gpu_count);
    if(cerr == cudaErrorNoDevice) {
        gpu_count = 0;
    } else {
        cudaCheck(cerr);
    }

    return gpu_count;
}