// https://stackoverflow.com/a/28996168

#include "barrier.h"

Barrier::Barrier(unsigned initialCount)
    : m_number_threads(initialCount)
    , m_count(initialCount)
{ }

void Barrier::wait()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    --m_count;
    if (m_count == 0)
    {
        m_cv.notify_all();
    }
    else
    {
        m_cv.wait(lock, [this]{ return m_count == 0; });
    }
}

void Barrier::reset()
{
    m_count = m_number_threads;
}