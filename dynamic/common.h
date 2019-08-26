#pragma once

#include <stdio.h>
#include "barrier.h"
#include "task.h"

#define N 10
#define CYCLES 10

static int id = 0;

int init_cuda();
void alloc_cuda(task_t* task);
void dealloc_cuda(task_t* task);

void* run_cuda(void* v_task);
void* run_openmp(void* v_task);


// int MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, MPI_Comm communicator, MPI_Status* status, int tags[], int tag_count)
// {
//     while(true) {
//         int flags[tag_count];
//         MPI_Status statuses[tag_count];
//         for(int i = 0; i < tag_count; i++) {
//             MPI_Iprobe(source, tags[i], communicator, &flags[i], &statuses[i]);
//         }

//         MPI_Waitany()
//     }

//     MPI_

//     MPI_Recv
// }