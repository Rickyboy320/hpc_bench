#include <vector>
#include <map>

#include <stdio.h>
#include "mpi.h"
#include "manager.h"

struct registration_t
{
    int start;
    int size;
    int rank;
    int deviceID;
};

static std::vector<registration_t*> registrations;
static std::map<int, int> device_map;

registration_t* find(int offset) {
    for (registration_t* x : registrations)
    {
        if(offset >= x->start && offset < x->start + x->size) {
            return x;
        }
    }

    throw std::exception();
}

bool is_available(int rank, int size) {
    int num_devices = device_map.find(rank)->second;
    for (registration_t* x : registrations)
    {
        if(x->rank == rank) {
            num_devices--;
        }
    }

    return num_devices > 0;
}

int find_available(int size, MPI_Comm &comm) {
    int world_size;
    MPI_Comm_size(comm, &world_size);

    for(int i = 0; i < world_size; i++) {
        if(is_available(i, size)) {
            return i;
        }
    }
}

void manage_nodes(void* v_comm)
{
    MPI_Comm* p_comm = (MPI_Comm*) v_comm;
    MPI_Comm manager_comm = *p_comm;

    printf("Hello world, comm: %p\n", &manager_comm);

    while(true) {
        MPI_Status status;

        int buffer;
        printf("Hello loop, comm: %p\n", &manager_comm);
        MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, manager_comm, &status);

        if(status.MPI_TAG == REGISTER) {
            int source = status.MPI_SOURCE;

            int deviceID;
            MPI_Recv(&deviceID, 1, MPI_INT, status.MPI_SOURCE, REGISTER, manager_comm, MPI_STATUS_IGNORE);

            int length;
            MPI_Recv(&length, 1, MPI_INT, status.MPI_SOURCE, REGISTER, manager_comm, MPI_STATUS_IGNORE);

            registration_t* reg = new registration_t();
            reg->start = buffer;
            reg->size = length;
            reg->rank = source;
            reg->deviceID = deviceID;
            registrations.push_back(reg);

            printf("Received registration. Rank %d, device: %d starts at %d, length: %d\n", source, deviceID, buffer, length);
        } else if(status.MPI_TAG == DEVICES) {
            int source = status.MPI_SOURCE;
            int num_devices = buffer;
            device_map.insert({source, num_devices});

            printf("Received device count. Rank %d has %d devices\n", source, num_devices);
        } else if(status.MPI_TAG == LOOKUP) {
            int source = status.MPI_SOURCE;
            int start = buffer;
            registration_t* registration = find(start);

            printf("Received and sent lookup. Rank %d requests node at %d. Found: %d: %d\n", source, start, registration->rank, registration->deviceID);
            int package[2] = {registration->rank, registration->deviceID};
            MPI_Send(package, 2, MPI_INT, source, LOOKUP, manager_comm);
        } else if(status.MPI_TAG == FREE) {
            int source = status.MPI_SOURCE;
            int size = buffer;
            int rank = find_available(size, manager_comm);
            printf("Received free node request. Rank %d requests buffer size of %d. Found: %d\n", source, size, rank);

            MPI_Send(&rank, 1, MPI_INT, source, FREE, manager_comm);
        } else if(status.MPI_TAG == TERMINATE) {
            pthread_exit(NULL);
        } else {
            printf("Received invalid tag: %d\n", status.MPI_TAG);
        }
    }
}