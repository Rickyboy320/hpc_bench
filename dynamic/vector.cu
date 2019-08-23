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
    int iteration = task->start_iteration;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Setting device: %d\n", task->cuda.id);
    cudaSetDevice(task->cuda.id);

    for(; iteration < CYCLES; iteration++) {
        // Copy the host input vectors A and B H2D.

        printf("A: %p, cudaA: %p, size: %d\n", task->cuda.A, &task->A[-1], task->cuda.size + 2*sizeof(float));

        int inset = 0;
        cudaCheck(cudaMemcpy(task->cuda.A, &task->A[-1], task->cuda.size + 2 * sizeof(float), cudaMemcpyHostToDevice));
        inset = 1;

        // Launch the vector-add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (task->size + threadsPerBlock - 1) / threadsPerBlock;

        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0>>>(task->cuda.A, task->cuda.C, task->size, inset);

        // Copy the device result vector D2H.
        cudaCheck(cudaMemcpy(task->C, task->cuda.C, task->cuda.size, cudaMemcpyDeviceToHost));

        cudaCheck(cudaDeviceSynchronize());

        printf("cuda wait\n");
        task->barrier->wait();

        // Switch buffers
        for(int j = 0; j < task->size; j++) {
            printf("C%d: (%d) %d: %f\n", iteration, rank, j, task->C[j]);

            task->A[j] = task->C[j];
        }

        printf("(%d) Updating neighbours\n", rank);
        std::vector<MPI_Request> requests;
        fetch_and_update_neighbours(rank, task, requests);
        // TODO: now this deadlocks because 3rd patch does not know that the patch changed.

        MPI_Status* statuses;
        MPI_Waitall(requests.size(), &requests[0], statuses);


        task->barrier->wait();
        //MPI Barrier @ mainthread
        task->barrier->wait();
    }

    printf("cuda done\n");
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