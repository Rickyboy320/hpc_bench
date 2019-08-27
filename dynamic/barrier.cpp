// https://stackoverflow.com/a/28996168
// https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11

#include "barrier.h"

Barrier::Barrier(unsigned initialCount)
    : m_number_threads(initialCount)
    , m_count(initialCount)
    , m_generation(0)
{ }

void Barrier::wait()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    m_count--;
    if (m_count <= 0)
    {
        m_generation++;
        m_count = m_number_threads;
        m_cv.notify_all();
    }
    else
    {
        unsigned generation = m_generation;
        m_cv.wait(lock, [this, generation]{ return m_generation != generation; });
    }
}

void Barrier::resize(int new_size) {
    std::unique_lock<std::mutex> lock(m_mtx);

    m_count = new_size - (m_number_threads - m_count);
    m_number_threads = new_size;

    lock.unlock();
}