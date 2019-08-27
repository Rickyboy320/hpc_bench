#pragma once

#include <stdio.h>
#include <mpi.h>
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

int MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, MPI_Comm communicator, MPI_Status* status, std::function<bool(int)> tag_matcher);
int MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, MPI_Comm communicator, MPI_Status* status, int tags[], int tag_count);
int MPI_Recv_all(std::vector<MPI_Receive_req> &receives, MPI_Comm communicator, MPI_Status* statuses);

int construct_tag(int device_id, bool next, int tag);
bool match_tag(int device_id, int next, int tag, int input);