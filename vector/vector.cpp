#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <sys/syscall.h>
#include <sys/types.h>

#define N 50

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

struct task_t {
    int size;
    float* A;
    float* B;
    float* C;
};


void init_openmp()
{
    omp_set_num_threads(omp_get_num_procs());
    printf("Number procs: %d\n", omp_get_num_procs());
}

void init_cuda()
{

}

int init_mpi()
{
    MPI_Init(NULL, NULL);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Node %s = rank %d\n", processor_name, world_rank);

    return world_rank;
}

void* run_openmp(void* v_task)
{
    task_t* task = (task_t*) v_task;
    float* A = task->A;
    float* B = task->B;
    float* C = task->C;
    int size = task->size;

    for(int i = 0; i < size; i++)
    {
        printf("%f %f %f\n", C[i], A[i], B[i]);
    }

    int i;
    #pragma omp parallel for private(i) shared(A,B,C)
    for (i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }

    printf("Evaluated size: %d\n", size);

    pthread_exit(NULL);
}

void* run_cuda(void* v_task)
{
    task_t* task = (task_t*) v_task;

    pthread_exit(NULL);
}

int main()
{
    float A[N*2];
    float B[N*2];
    float C[N*2];

    int i;
    struct timeval tv1, tv2;
    struct timezone tz;
	double elapsed;

    int rank = init_mpi();

    // Init array

    int start = rank == 0 ? 0 : N;
    int receive = rank == 0 ? N : 0;

    for (i = start; i < start + N; i++)
    {
        A[i] = rank + 1;
        B[i] = rank + 1;
    }

    // Synchronize halfs of array with other node.

    MPI_Request requests[2];
    MPI_Irecv(&A[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&B[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &requests[1]);

    MPI_Send(&A[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
    MPI_Send(&B[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);

    MPI_Status statusses[2];
    MPI_Waitall(2, requests, statusses);


    //  Sync for 'equal' starts.
    MPI_Barrier(MPI_COMM_WORLD);


    // Distribute tasks evenly over CPU (OpenMP) and GPUs (CUDA)
    int gpu_count;
    cudaError_t cerr = cudaGetDeviceCount(&gpu_count);
    if(cerr == cudaErrorNoDevice) {
        gpu_count = 0;
    } else {
        cudaCheck(cerr);
    }

    printf("Count GPU devices: %d\n", gpu_count);

    pthread_t threads[gpu_count + 1];
    task_t tasks[gpu_count + 1];

    int sizePerDevice = ceil(N / (gpu_count + 1.0));
    for(int i = 0; i < gpu_count + 1; i++)
    {
        tasks[i].A = &A[start + sizePerDevice * i];
        tasks[i].B = &B[start + sizePerDevice * i];
        tasks[i].C = &C[start + sizePerDevice * i];

        tasks[i].size = sizePerDevice;
    }


    // Start benchmark
    gettimeofday(&tv1, &tz);



    int err = pthread_create(&threads[0], NULL, run_openmp, &tasks[0]);
    if (err != 0)
    {
        printf("Error on create: %d\n", err);
    }

    for(int i = 1; i <= gpu_count; i++)
    {
        int err = pthread_create(&threads[i], NULL, run_cuda, &tasks[i]);
        if (err != 0)
        {
            printf("Error on create: %d\n", err);
        }
    }


    for(int i = 0; i <= gpu_count; i++)
    {
        pthread_join(threads[i], NULL);
    }

//  Sync to wait on all processes.
    MPI_Barrier(MPI_COMM_WORLD);



    gettimeofday(&tv2, &tz);


    MPI_Request request;
    MPI_Irecv(&C[receive], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD, &request);
    MPI_Send(&C[start], N, MPI_FLOAT, (rank + 1) % 2, 0, MPI_COMM_WORLD);

    MPI_Status status;
    MPI_Wait(&request, &status);

    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    if(rank == 0)
    {
        printf("elapsed time = %f seconds.\n", elapsed);

        for (i = 0; i < 2*N; i++)
        {
            printf("%d: %f\n", i, C[i]);
        }
    }

    MPI_Finalize();
}