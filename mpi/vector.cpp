#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "variants.h"
#include "common.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Node %s = rank %d\n", processor_name, world_rank);

    MPI_Info info;
    MPI_Info_create(&info);

    //MPI_Info_set(info, "add-host", "node052");

    MPI_Comm parentcomm, intercomm;
    MPI_Comm_get_parent( &parentcomm );
    if (parentcomm == MPI_COMM_NULL) {
        int sub_processes = 1;
        printf("Rank %d spawning %d proces(ses)\n", world_rank, sub_processes);

        int errcodes[sub_processes];
        if(argc > 1) {
            MPI_Comm_spawn(argv[0], &argv[1], sub_processes, info, 0, MPI_COMM_WORLD, &intercomm, errcodes );
        } else {
            MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, sub_processes, info, 0, MPI_COMM_WORLD, &intercomm, errcodes );
        }
    }
    else
    {
        printf("Rank %d is a child (own communicator!).\n", world_rank);
    }

    MPI_Finalize();
}
