#include <map>

#include "mpi.h"
#include "manager.h"

static std::map<int, int> registration_map;
static std::map<int, int> device_map;

bool is_available(int rank, int size) {
    int num_devices = device_map.find(rank)->second;
    for (auto const& x : registration_map)
    {
        if(x.second == rank) {
            num_devices--;
        }
    }

    return num_devices > 0;
}

int find_available(int size) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for(int i = 0; i < world_size; i++) {
        if(is_available(i, size)) {
            return i;
        }
    }
}

void manage_nodes()
{
    while(true) {
        MPI_Status status;

        int buffer;
        MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if(status.MPI_TAG == REGISTER) {
            int source = status.MPI_SOURCE;
            int start = buffer;
            registration_map.insert({start, source});

            printf("Received registration. Rank %d starts at %d\n", source, start);
        } else if(status.MPI_TAG == DEVICES) {
            int source = status.MPI_SOURCE;
            int num_devices = buffer;
            device_map.insert({source, num_devices});

            printf("Received device count. Rank %d has %d devices\n", source, num_devices);
        } else if(status.MPI_TAG == LOOKUP) {
            int source = status.MPI_SOURCE;
            int start = buffer;
            auto rank = registration_map.find(start);

            printf("Received and sent lookup. Rank %d requests node at %d. Found: %d\n", source, start, rank);

            MPI_Send(&rank->second, 1, MPI_INT, source, LOOKUP, MPI_COMM_WORLD);

        } else if(status.MPI_TAG == FREE) {
            int source = status.MPI_SOURCE;
            int size = buffer;
            int rank = find_available(size);
            printf("Received free node request. Rank %d requests buffer size of %d. Found: %d\n", source, size, rank);

            MPI_Send(&rank, 1, MPI_INT, source, FREE, MPI_COMM_WORLD);
        } else if(status.MPI_TAG == TERMINATE) {
            pthread_exit(NULL);
        }
    }
}