#include <vector>
#include <map>

#include <stdio.h>
#include "mpi.h"
#include "manager.h"
#include "common.h"

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

    throw std::runtime_error("No registration found containing offset.\n");
}

registration_t* get_registration(int rank, int deviceID) {
    for (registration_t* x : registrations)
    {
        if(x->rank == rank && x->deviceID == deviceID) {
            return x;
        }
    }

    throw std::runtime_error("No registration found for rank and device id.\n");
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

    while(true) {
        MPI_Status status;

        int buffer;
        MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, manager_comm, &status, [](int tag) {
            for(int i = 0; i < MANAGER_TAGS_LENGTH; i++) {
                if(match_tag(-1, -1, MANAGER_TAGS[i], tag)) {
                    return true;
                }
            }
            return false;
        });

        if(match_tag(-1, -1, REGISTER, status.MPI_TAG)) {
            int source = status.MPI_SOURCE;

            int deviceID;
            MPI_Recv(&deviceID, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, manager_comm, MPI_STATUS_IGNORE);

            int length;
            MPI_Recv(&length, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, manager_comm, MPI_STATUS_IGNORE);

            registration_t* reg = new registration_t();
            reg->start = buffer;
            reg->size = length;
            reg->rank = source;
            reg->deviceID = deviceID;
            registrations.push_back(reg);

            printf("Received registration. Rank %d, device: %d starts at %d, length: %d\n", source, deviceID, buffer, length);
        } else if(match_tag(-1, -1, DEVICES, status.MPI_TAG)) {
            int source = status.MPI_SOURCE;
            int num_devices = buffer;
            device_map.insert({source, num_devices});

            printf("Received device count. Rank %d has %d devices\n", source, num_devices);
        } else if(match_tag(-1, -1, UPDATE, status.MPI_TAG)) {
            int source = status.MPI_SOURCE;
            int deviceID = buffer;
            int size;
            MPI_Recv(&size, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, manager_comm, MPI_STATUS_IGNORE);
            registration_t* reg = get_registration(source, deviceID);
            reg->size = size;

            printf("Received update. Rank %d, device %d now has length: %d\n", source, deviceID, size);
        } else if(match_tag(-1, -1, LOOKUP, status.MPI_TAG)) {
            int source = status.MPI_SOURCE;
            int start = buffer;
            registration_t* registration = find(start);

            printf("Received and sent lookup. Rank %d requests node at %d. Found: %d: %d\n", source, start, registration->rank, registration->deviceID);
            int package[2] = {registration->rank, registration->deviceID};
            MPI_Send(package, 2, MPI_INT, source, status.MPI_TAG, manager_comm);
        } else if(match_tag(-1, -1, FREE, status.MPI_TAG)) {
            int source = status.MPI_SOURCE;
            int size = buffer;
            int rank = find_available(size, manager_comm);
            printf("Received free node request. Rank %d requests buffer size of %d. Found: %d\n", source, size, rank);

            MPI_Send(&rank, 1, MPI_INT, source, status.MPI_TAG, manager_comm);
        } else if(match_tag(-1, -1, TERMINATE, status.MPI_TAG)) {
            pthread_exit(NULL);
        } else {
            printf("Received invalid tag: %d\n", status.MPI_TAG);
        }
    }
}