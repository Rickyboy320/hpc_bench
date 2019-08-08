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

void run_cuda_stream(task_t task, cudaStream_t stream)
{
    int size = task.size;
    size_t byteSize = size * sizeof(float);

    // Allocate the device vectors
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaCheck(cudaHostRegister(task.A, byteSize, 0));
    cudaCheck(cudaHostRegister(task.B, byteSize, 0));
    cudaCheck(cudaHostRegister(task.C, byteSize, 0));

    cudaCheck(cudaMalloc((void **)&d_A, byteSize));
    cudaCheck(cudaMalloc((void **)&d_B, byteSize));
    cudaCheck(cudaMalloc((void **)&d_C, byteSize));

    // Copy the host input vectors A and B H2D.
    cudaCheck(cudaMemcpyAsync(d_A, task.A, byteSize, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_B, task.B, byteSize, cudaMemcpyHostToDevice, stream));

    // Launch the vector-add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, size);

    cudaCheck(cudaStreamSynchronize(stream));

    // Copy the device result vector D2H.
    cudaCheck(cudaMemcpyAsync(task.C, d_C, byteSize, cudaMemcpyDeviceToHost, stream));

    // Free device global memory
    cudaCheck(cudaHostUnregister(task.A));
    cudaCheck(cudaHostUnregister(task.B));
    cudaCheck(cudaHostUnregister(task.C));

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));

}

void* run_cuda(void* v_task)
{
    task_t* task = (task_t*) v_task;

    run_cuda_stream(*task, cudaStreamPerThread);

    if(task->is_threads)
    {
        pthread_exit(NULL);
    }
    else
    {
        return NULL;
    }
}

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

void sync_cuda_streams(int gpu_count, cudaStream_t* streams)
{
    for(int i = 0; i < gpu_count; i++)
    {
        cudaCheck(cudaStreamDestroy(streams[i]));
    }

    free(streams);
}

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