#pragma once

#include <vector>
#include <mpi.h>
#include <thread>
#include "barrier.h"
#include "task.h"

struct manager_info_t {
    Barrier* barrier;
    Barrier* start_barrier;
    int* iteration;
    MPI_Comm* manager;

    std::vector<task_t>* tasks;
    std::vector<std::thread>* threads;
};

void listen_split(void* v_info);