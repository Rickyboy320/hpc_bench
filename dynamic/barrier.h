// https://stackoverflow.com/a/28996168
// https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11

#pragma once

#include <mutex>
#include <condition_variable>

class Barrier
{
private:
    std::mutex m_mtx;
    std::condition_variable m_cv;
    unsigned m_count;
    unsigned m_number_threads;
    unsigned m_generation;

public:
    Barrier(unsigned initialCount);

    void wait();
    void expand();
    void shrink();
    void resize(int new_size);

    int get_size();

private:
    void resize_barrier(int new_size);
};