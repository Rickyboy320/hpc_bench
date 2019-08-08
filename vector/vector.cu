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
vectorAdd(const float *A, const float *B, float *C, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        C[i] = A[i] + B[i];
    }
}

// Allocate cuda memory and pin host memory (required for async stream).
void alloc_cuda(task_t* task)
{
    int size = task->size;

    task->cudamem.size = size * sizeof(float);

    // Allocate the device vectors
    cudaCheck(cudaHostRegister(task->A, task->cudamem.size, 0));
    cudaCheck(cudaHostRegister(task->B, task->cudamem.size, 0));
    cudaCheck(cudaHostRegister(task->C, task->cudamem.size, 0));

    cudaCheck(cudaMalloc((void **)&task->cudamem.A, task->cudamem.size));
    cudaCheck(cudaMalloc((void **)&task->cudamem.B, task->cudamem.size));
    cudaCheck(cudaMalloc((void **)&task->cudamem.C, task->cudamem.size));
}

// Deallocate cuda memory and unpin host memory.
void dealloc_cuda(task_t* task)
{
    // Free device global memory
    cudaCheck(cudaHostUnregister(task->A));
    cudaCheck(cudaHostUnregister(task->B));
    cudaCheck(cudaHostUnregister(task->C));

    cudaCheck(cudaFree(task->cudamem.A));
    cudaCheck(cudaFree(task->cudamem.B));
    cudaCheck(cudaFree(task->cudamem.C));
}

// Run cuda kernel asynchronously on the given stream.
void run_cuda_stream(task_t task, cudaStream_t stream)
{
    // Copy the host input vectors A and B H2D.
    cudaCheck(cudaMemcpyAsync(task.cudamem.A, task.A, task.cudamem.size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(task.cudamem.B, task.B, task.cudamem.size, cudaMemcpyHostToDevice, stream));

    // Launch the vector-add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (task.size + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(task.cudamem.A, task.cudamem.B, task.cudamem.C, task.size);

    // Copy the device result vector D2H.
    cudaCheck(cudaMemcpyAsync(task.C, task.cudamem.C, task.cudamem.size, cudaMemcpyDeviceToHost, stream));
}

// Run the cuda task (on the 'thread stream').
void* run_cuda(void* v_task)
{
    task_t* task = (task_t*) v_task;

    run_cuda_stream(*task, cudaStreamPerThread);
    cudaCheck(cudaStreamSynchronize(cudaStreamPerThread));

    if(task->is_threads)
    {
        pthread_exit(NULL);
    }
    else
    {
        return NULL;
    }
}

// Create and run streams for each of the tasks.
cudaStream_t* run_cuda_streams(int gpu_count, task_t tasks[])
{
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * gpu_count);
    for (int i = 0; i < gpu_count; i++)
    {
        cudaCheck(cudaStreamCreate(&streams[i]));

        run_cuda_stream(tasks[i + 1], streams[i]);
    }

    return streams;
}

// Syncrhonize and delete all streams.
void sync_cuda_streams(int gpu_count, cudaStream_t* streams)
{
    for(int i = 0; i < gpu_count; i++)
    {
        cudaCheck(cudaStreamSynchronize(streams[i]));
        cudaCheck(cudaStreamDestroy(streams[i]));
    }

    free(streams);
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