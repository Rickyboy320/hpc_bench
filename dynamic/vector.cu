#include <pthread.h>

#include "manager.h"
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

    // Allocate the device vectors
    cudaCheck(cudaMalloc((void **)&task->cuda.A, (task->size + 2) * sizeof(float))); // Plus 'imported' neighbours.
    cudaCheck(cudaMalloc((void **)&task->cuda.C, task->size * sizeof(float)));
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

        printf("A: %p, cudaA: %p, size: %d\n", task->cuda.A, &task->A[-1], (task->size + 2) * sizeof(float));

        int inset = 0;
        cudaCheck(cudaMemcpy(task->cuda.A, &task->A[-1], (task->size + 2) * sizeof(float), cudaMemcpyHostToDevice));
        inset = 1;

        // Launch the vector-add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (task->size + threadsPerBlock - 1) / threadsPerBlock;

        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0>>>(task->cuda.A, task->cuda.C, task->size, inset);

        // Copy the device result vector D2H.
        cudaCheck(cudaMemcpy(task->C, task->cuda.C, task->size * sizeof(float), cudaMemcpyDeviceToHost));

        cudaCheck(cudaDeviceSynchronize());

        printf("cuda wait\n");
        task->barrier->wait();

        // Switch buffers
        for(int j = 0; j < task->size; j++) {
            printf("C%d: (%d) [%d] %d: %f\n", iteration, rank, task->id, j, task->C[j]);

            task->A[j] = task->C[j];
        }

        printf("(%d) Updating neighbours\n", rank);
        std::vector<MPI_Receive_req> requests;
        std::vector<int> types;
        fetch_and_update_neighbours(rank, task, requests, types, false);

        // Split
        // if(will_split) {
        //      // Arbitrarily (as a test) decide to split.
        //      split(task, rank, target);
        // }

        MPI_Status statuses[requests.size()];
        if(!requests.empty()) {
            MPI_Recv_all(requests, MPI_COMM_WORLD, statuses);
        }

        for(int j = -1; j < task->size + 1; j++) {
            printf("A @ C%d: (%d) [%d] %d: %f\n", iteration, rank, task->id, j, task->A[j]);
        }

        task->barrier->wait();
        // MPI wait
        task->barrier->wait();
        for(int i = 0; i < requests.size(); i++) {
            printf("Checking for split: %d\n", statuses[i].MPI_TAG);
            if(match_tag(-1, -1, SPLIT, statuses[i].MPI_TAG)) {
                // Received notification of split of target. Will update refs.
                if(types[i] == NEXT_TYPE) {
                    printf("(%d:%d) Update nextref\n", rank, task->id);
                    int start = task->offset + task->size;
                    MPI_Send(&start, 1, MPI_INT, MANAGER_RANK, LOOKUP, *task->manager);
                    int package[2];
                    MPI_Recv(&package, 2, MPI_INT, MANAGER_RANK, LOOKUP, *task->manager, MPI_STATUS_IGNORE);
                    task->next.rank = package[0];
                    task->next.id = package[1];
                } else if(types[i] == PREV_TYPE) {
                    printf("(%d:%d) Update prevref\n", rank, task->id);
                    int start = task->offset - 1;
                    MPI_Send(&start, 1, MPI_INT, MANAGER_RANK, LOOKUP, *task->manager);
                    int package[2];
                    MPI_Recv(&package, 2, MPI_INT, MANAGER_RANK, LOOKUP, *task->manager, MPI_STATUS_IGNORE);
                    task->prev.rank = package[0];
                    task->prev.id = package[1];
                } else {
                    throw std::runtime_error("Invalid SPLIT type received.");
                }
            }
        }
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