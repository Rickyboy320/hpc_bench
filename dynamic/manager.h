#pragma once

#include <mpi.h>

static const int REGISTER = 10000;
static const int LOOKUP = 10001;
static const int FREE = 10002;
static const int DEVICES = 10003;
static const int TERMINATE = 10004;

static const int SPLIT = 5000;

static const int MANAGER_RANK = 0;

void manage_nodes(void* v_comm);