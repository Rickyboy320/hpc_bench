#pragma once

#include "common.h"

enum Variant { mpi_spawn, mpi_idle, mpi_single };

void run_spawn_variant(int argc, char** argv, int rank, task_t task);
void run_idle_variant(int rank, int world_size, task_t task);

void run_single_variant(task_t task);
