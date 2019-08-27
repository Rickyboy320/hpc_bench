#include "common.h"

int MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, MPI_Comm communicator, MPI_Status* status, std::function<bool(int)> tag_matcher)
{
    while(true) {
        MPI_Status probe_status;
        MPI_Probe(source, MPI_ANY_TAG, communicator, &probe_status);

        if(tag_matcher(probe_status.MPI_TAG)) {
            return MPI_Recv(buffer, count, datatype, source, probe_status.MPI_TAG, communicator, status);
        }
    }
}

int MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, MPI_Comm communicator, MPI_Status* status, int tags[], int tag_count)
{
    return MPI_Recv(buffer, count, datatype, source, communicator, status, [tags, tag_count](int tag) {
        for(int i = 0; i < tag_count; i++) {
            if(tag == tags[i]) {
                return true;
            }
        }
        return false;
    });
}

int MPI_Recv_all(std::vector<MPI_Receive_req> &receives, MPI_Comm communicator, MPI_Status* statuses)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int completed = 0;
    while(completed < receives.size()) {
        MPI_Status probe_status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator, &probe_status);

        printf("(%d) Found message from %d tag %d\n", rank, probe_status.MPI_SOURCE, probe_status.MPI_TAG);

        for(int i = 0; i < receives.size(); i++)
        {
            if(!receives[i].completed && probe_status.MPI_SOURCE == receives[i].source)
            {
                printf("Matching tags..\n");
                if(receives[i].tag_matcher(probe_status.MPI_TAG))
                {
                    printf("(%d) Matched. Now receiving.\n", rank);

                    receives[i].completed = true;
                    completed++;

                    int error;
                    if(statuses == MPI_STATUSES_IGNORE) {
                        MPI_Recv(receives[i].buffer, receives[i].count, receives[i].datatype, probe_status.MPI_SOURCE, probe_status.MPI_TAG, communicator, MPI_STATUS_IGNORE);
                    } else {
                        MPI_Recv(receives[i].buffer, receives[i].count, receives[i].datatype, probe_status.MPI_SOURCE, probe_status.MPI_TAG, communicator, &statuses[i]);
                    }

                    if(error != MPI_SUCCESS)
                    {
                        return error;
                    }

                    break;
                }
            }
        }
    }
    return MPI_SUCCESS;
}


int construct_tag(int device_id, bool next, int tag)
{
    if(!(next == 0 || next == 1)) {
        throw std::runtime_error("Invalid tag. Next should be either 0 or 1.");
    }

    if(tag > 999) {
        throw std::runtime_error("Invalid tag. Tag should be less than 1000.");
    }

    printf("Input: %d, %d, %d. Output: %d\n", device_id, next, tag, device_id * 10000 + tag + next);

    return device_id * 10000 + tag * 10 + next;
}

bool match_tag(int device_id, int next, int tag, int input) {
    if(device_id != -1) {
        if(input / 10000 != device_id) {
            printf("Invalid device. Input: %d, device: %d, match: %d\n", input, device_id, input / 10000);
            return false;
        }
    }

    if(next != -1) {
        if(input % 2 != next) {
            printf("Invalid next\n");
            return false;
        }
    }

    if(tag != -1) {
        if((input % 10000) / 10 != tag) {
            printf("Invalid tag\n");
            return false;
        }
    }

    return true;
}