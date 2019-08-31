#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>


#include "task.h"
#include "common.h"
#include "barrier.h"
#include "manager.h"
#include "devicemanager.h"

int init()
{
    omp_set_num_threads(omp_get_num_procs());
    printf("Number omp procs: %d\n", omp_get_num_procs());

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
    // Unpack task
    task_t* task = (task_t*) v_task;
    float* A = task->A;
    float* C = task->C;
    int iteration = task->start_iteration;
    int rank;
    MPI_Comm manager_comm = *task->manager;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("(%d:%d) Waiting start omp barrier: size: %d\n", rank, task->id, task->start_barrier->get_size());
    task->start_barrier->wait();

    for(; iteration < CYCLES; iteration++) {
        int size = task->size;

        printf("(%d:%d) Task offset: %d, size: %d\n", rank, task->id, task->offset, task->size);

        // Run task (sum neighbours) with OpenMP
        int i;
        #pragma omp parallel for private(i) shared(A,C)
        for (i = 0; i < size; i++)
        {
            C[i] = A[i] + A[i-1] + A[i+1];
            printf("A: %f A-1 %f A+1 %f C: %f\n", A[i], A[i-1], A[i+1], C[i]);
        }

        printf("omp barrier\n");
        task->barrier->wait();

        //Switch buffers
        for(int j = 0; j < size; j++) {
            printf("C%d: (%d) [%d] %d: %f\n", iteration, rank, task->id, j, C[j]);

            A[j] = C[j];
        }

        // Arbitrarily decide to split
        bool will_split = (iteration == 3 && rank == 0) || (iteration == 8 && rank == 1) || (iteration == 9 && rank == 2);

        printf("(%d) Updating neighbours\n", rank);

        std::vector<MPI_Receive_req> requests;
        std::vector<int> types;
        fetch_and_update_neighbours(rank, task, requests, types, will_split);

        MPI_Status statuses[requests.size()];
        if(!requests.empty()) {
            MPI_Recv_all(requests, MPI_COMM_WORLD, statuses);
        }

        for(int j = -1; j < size + 1; j++) {
            printf("A @ C%d: (%d) [%d] %d: %f\n", iteration, rank, task->id, j, A[j]);
        }

        // Split
        if(will_split) {
             split(task, rank);
        }

        task->barrier->wait();
        // MPI barrier
        task->barrier->wait();

        for(int i = 0; i < requests.size(); i++) {
            if(match_tag(-1, -1, WILL_SPLIT, statuses[i].MPI_TAG)) {
                int tag = construct_tag(task->id, 0, LOOKUP);

                // Received notification of split of target. Will update refs.
                if(types[i] == NEXT_TYPE) {
                    printf("(%d:%d) Update nextref\n", rank, id);
                    int start = task->offset + task->size;
                    MPI_Send(&start, 1, MPI_INT, MANAGER_RANK, tag, *task->manager);
                    int package[2];
                    MPI_Recv(&package, 2, MPI_INT, MANAGER_RANK, tag, *task->manager, MPI_STATUS_IGNORE);
                    task->next.rank = package[0];
                    task->next.id = package[1];
                    printf("(%d:%d) New next: %d:%d\n",  rank, task->id, task->next.rank, task->next.id);
                } else if(types[i] == PREV_TYPE) {
                    printf("(%d:%d) Update prevref\n", rank, id);
                    int start = task->offset - 1;
                    MPI_Send(&start, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);
                    int package[2];
                    MPI_Recv(&package, 2, MPI_INT, MANAGER_RANK, tag, manager_comm, MPI_STATUS_IGNORE);
                    task->prev.rank = package[0];
                    task->prev.id = package[1];
                    printf("(%d:%d) New prev: %d:%d\n",  rank, task->id, task->prev.rank, task->prev.id);
                } else {
                    throw std::runtime_error("Invalid SPLIT type received.");
                }
            }
        }

        printf("(%d:%d) Waiting endsdtart omp barrier: size: %d\n", rank, task->id, task->start_barrier->get_size());
        task->start_barrier->wait();
    }

    printf("omp done\n");
    pthread_exit(NULL);
}

void run_cthread_variant(int rank, std::vector<task_t*> &tasks, Barrier* barrier, Barrier* start_barrier, MPI_Comm& manager, std::thread* device_thread, std::vector<std::thread>& threads, int* iteration)
{
    // Pre communication: fill input arrays with neighbouring data.
    std::vector<MPI_Receive_req> requests;
    std::vector<int> types;
    for(int i = 0; i < tasks.size(); i++) {
        fetch_and_update_neighbours(rank, tasks[i], requests, types, false);
    }

    printf("(%d) Pre Waiting all\n", rank);
    if(!requests.empty()) {
        MPI_Recv_all(requests, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }

    printf("(%d) Starting threads: %d\n", rank, tasks.size());
    for(int i = 0; i < tasks.size(); i++) {
        if(tasks[i]->type == CPU) {
            threads.push_back(std::thread(run_openmp, tasks[i]));
        } else if(tasks[i]->type == GPU) {
            threads.push_back(std::thread(run_cuda, tasks[i]));
        } else {
            printf("WARNING: task without type: %p\n", &tasks[i]);
            throw std::runtime_error("Task without type.");
        }
    }

    printf("(%d) Waiting start barrier: size: %d\n", rank, start_barrier->get_size());
    start_barrier->wait();

    for(; *iteration < CYCLES; (*iteration)++) {
        printf("(%d) Waiting barrier main\n", rank);
        barrier->wait();

        // Devices switch buffers (on-site)

        // Devices fetch neighbours (on-site)

        //  Sync to wait on all processes.
        barrier->wait();

        printf("(%d) Waiting MPI\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
        barrier->wait();

        // Update was made in the start barrier, now is a safe time to update the normal barrier.
        if(barrier->get_size() != start_barrier->get_size()) {
            printf("Updating barrier size. Was: %d, now: %d\n", barrier->get_size(), start_barrier->get_size());
            barrier->resize(start_barrier->get_size());
        }

        printf("(%d) Waiting start barrier: size: %d\n", rank, start_barrier->get_size());
        start_barrier->wait();
    }

    // Wait for all tasks to complete.
    for(int i = 0; i < threads.size(); i++)
    {
        printf("(%d) Joining %d\n", rank, i);
        threads[i].join();
    }
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("ERROR: The MPI library does not have full thread support. Provided: %d\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = init();

    // Parse command line args

    int active_devices = 1;
    if(argc > 1) {
        active_devices = std::stoi(argv[1]);
    }

    // Distribute tasks evenly over nodes and devices.
    int gpu_count = init_cuda();
    int task_count = rank < active_devices ? gpu_count + 1 : 0;

    printf("Rank: %d: Count GPU devices: %d. Tasks: %d\n", rank, gpu_count, task_count);

    Barrier barrier(task_count + 1);
    Barrier start_barrier(task_count + 1);

    MPI_Comm manager_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &manager_comm);
    printf("(%d): Manager comm after split: %p\n", rank, &manager_comm);

    std::vector<task_t*> tasks;
    if(task_count > 0) {
        init_tasks(tasks, task_count, &barrier, &start_barrier, &manager_comm, active_devices);
    }

    std::thread manage_thread;
    if(rank == MANAGER_RANK) {
        manage_thread = std::thread(manage_nodes, &manager_comm);
    }

    // Register devices
    int device_count = 1 + gpu_count;
    fprintf(stderr, "(%d) Sending device info: %d.\n", rank, device_count);
    MPI_Send(&device_count, 1, MPI_INT, MANAGER_RANK, construct_tag(0, 0, DEVICES), manager_comm);

    // Register tasks
    for(int i = 0; i < tasks.size(); i++) {
        int tag = construct_tag(i, 0, REGISTER);
        fprintf(stderr, "(%d) Sending task info.\n", rank);
        MPI_Send(&tasks[i]->offset, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);
        MPI_Send(&tasks[i]->id, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);
        MPI_Send(&tasks[i]->size, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);

        fprintf(stderr, "sent task info.\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i < tasks.size(); i++) {
        int tag = construct_tag(i, 0, LOOKUP);
        if(i == 0 && rank == 0) {
            tasks[i]->prev.rank = -1;
        } else {
            int lookup = tasks[i]->offset - 1;
            MPI_Send(&lookup, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);

            int receive[2];
            MPI_Recv(&receive, 2, MPI_INT, MANAGER_RANK, tag, manager_comm, MPI_STATUS_IGNORE);

            tasks[i]->prev.rank = receive[0];
            tasks[i]->prev.id = receive[1];
        }
        if(i == task_count - 1 && rank == active_devices - 1) {
            tasks[i]->next.rank = -1;
        } else {
            int lookup = tasks[i]->offset + tasks[i]->size;
            MPI_Send(&lookup, 1, MPI_INT, MANAGER_RANK, tag, manager_comm);

            int receive[2];
            MPI_Recv(&receive, 2, MPI_INT, MANAGER_RANK, tag, manager_comm, MPI_STATUS_IGNORE);

            tasks[i]->next.rank = receive[0];
            tasks[i]->next.id = receive[1];
        }
    }

    int iteration = 0;
    std::vector<std::thread> threads;

    manager_info_t info;
    info.barrier = &barrier;
    info.start_barrier = &start_barrier;
    info.iteration = &iteration;
    info.tasks = &tasks;
    info.threads = &threads;
    info.manager = &manager_comm;

    std::thread device_thread = std::thread(listen_split, &info);

    //  Sync for 'equal' starts.
    fprintf(stderr, "(%d) Barrier\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stderr, "(%d) Starting\n", rank);

    // Run tasks
    run_cthread_variant(rank, tasks, &barrier, &start_barrier, manager_comm, &device_thread, threads, &iteration);

    for(int i = 0; i < tasks.size(); i++)
    {
        if(tasks[i]->type == GPU) {
            dealloc_cuda(tasks[i]);
        }
    }

    // Communicate result over MPI & verify.
    for(int i = 0; i < tasks.size(); i++) {
        for(int j = 0; j < tasks[i]->size; j++) {
            printf("(%d) [%d] %d: %f\n", rank, i, j, tasks[i]->C[j]);
        }

        free(tasks[i]);
    }

    int empty = 0;
    MPI_Send(&empty, 1, MPI_INT, rank, construct_tag(0, 0, TERMINATE), manager_comm);

    if(rank == MANAGER_RANK)
    {
        MPI_Send(&empty, 1, MPI_INT, MANAGER_RANK, construct_tag(0, 0, TERMINATE), manager_comm);
        manage_thread.join();
    }

    device_thread.join();

    MPI_Finalize();
}