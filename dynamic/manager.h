#pragma once

#include <mpi.h>

static const int REGISTER = 0;
static const int LOOKUP = 1;
static const int FREE = 2;
static const int DEVICES = 3;
static const int TERMINATE = 4;

static int MANAGER_TAGS_LENGTH = 5;
static int MANAGER_TAGS[] = {REGISTER, LOOKUP, FREE, DEVICES, TERMINATE};

static const int SPLIT = 10;

static const int MANAGER_RANK = 0;

void manage_nodes(void* v_comm);