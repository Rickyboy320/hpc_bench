#include <map>
#include <thread>
#include <mpi.h>

#include "devicemanager.h"
#include "task.h"
#include "manager.h"
#include "common.h"

void listen_split(void* v_info)
{
    manager_info_t* info = (manager_info_t*) v_info;

    Barrier* barrier = info->barrier;
    Barrier* start_barrier = info->start_barrier;
    std::vector<task_t>& tasks = *info->tasks;
    std::vector<std::thread>& threads = *info->threads;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while(true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, *info->manager, &status);

        if(status.MPI_TAG == SPLIT) {
            start_barrier->expand();
            receive_split(rank, status.MPI_SOURCE, tasks, *info->manager);
            int index = tasks.size() - 1;
            tasks[index].type = CPU;
            tasks[index].start_iteration = *info->iteration + 1;
            tasks[index].barrier = barrier;
            tasks[index].start_barrier = start_barrier;
            tasks[index].manager = info->manager;

            threads.push_back(std::thread(run_openmp, &tasks[tasks.size() - 1]));
        } else if(status.MPI_TAG == TERMINATE) {
            int buffer;
            MPI_Recv(&buffer, 1, MPI_INT, status.MPI_SOURCE, TERMINATE, *info->manager, &status);
            pthread_exit(NULL);
        }
    }
}