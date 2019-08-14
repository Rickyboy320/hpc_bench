Benchmark:

Idle:
- On one processor;
    - Main process
    - 15 idle processes
- Investigate runtime influence of the 15 idle processes (do they slow down compared to when there are no idles?)

Spawn:
- On one processor;
    - Main process
    - 0 idle processes, spawn 15 when needed.
- Investigate runtime 'offset' - how long does it take for the spawned to 'catch up' (likely program dependent).
