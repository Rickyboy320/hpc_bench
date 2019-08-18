// https://stackoverflow.com/a/28996168

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

public:
    Barrier(unsigned initialCount);

    void wait();

    void reset();
};