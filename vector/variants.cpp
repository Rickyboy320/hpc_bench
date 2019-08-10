#include <thread>
#include <future>
#include <pthread.h>
#include <omp.h>

#include "common.h"

void run_pthread_variant(int rank, int gpu_count, task_t tasks[])
{
    pthread_t threads[gpu_count + 1];

    for(int i = 0; i < gpu_count + 1; i++)
    {
        tasks[i].is_threads = true;
    }

    int err = pthread_create(&threads[0], NULL, run_openmp, &tasks[0]);
    if (err != 0)
    {
        printf("Rank: %d: Error on create: %d\n", rank, err);
    }

    for(int i = 1; i <= gpu_count; i++)
    {
        int err = pthread_create(&threads[i], NULL, run_cuda, &tasks[i]);
        if (err != 0)
        {
            printf("Rank: %d: Error on create: %d\n", rank, err);
        }
    }

    // Wait for all tasks to complete.
    for(int i = 0; i <= gpu_count; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

void run_cthread_variant(int rank, int gpu_count, task_t tasks[])
{
    std::thread threads[gpu_count + 1];

    for(int i = 0; i < gpu_count + 1; i++)
    {
        tasks[i].is_threads = true;
    }

    threads[0] = std::thread(run_openmp, &tasks[0]);

    for(int i = 1; i <= gpu_count; i++)
    {
        threads[i] = std::thread(run_cuda, &tasks[i]);
    }

    // Wait for all tasks to complete.
    for(int i = 0; i <= gpu_count; i++)
    {
        threads[i].join();
    }
}

void run_async_variant(int rank, int gpu_count, task_t tasks[])
{
    std::future<void*> futures[gpu_count + 1];

    futures[0] = std::async(run_openmp, &tasks[0]);

    for(int i = 1; i <= gpu_count; i++)
    {
        futures[i] = std::async(run_cuda, &tasks[i]);
    }

    // Wait for all tasks to complete.
    for(int i = 0; i <= gpu_count; i++)
    {
        futures[i].wait();
    }
}

void run_openmp_variant(int rank, int gpu_count, task_t tasks[])
{
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                run_openmp(&tasks[0]);
            }

            for(int i = 1; i <= gpu_count; i++)
            {
                #pragma omp task
                {
                    run_cuda(&tasks[i]);
                }
            }
        }

        #pragma omp taskwait
    }
}

void run_cuda_variant(int rank, int gpu_count, task_t tasks[])
{
    cudaStream_t* streams = run_cuda_streams(gpu_count, tasks);

    run_openmp(&tasks[0]);

    sync_cuda_streams(gpu_count, streams);
}