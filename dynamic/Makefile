CC = mpic++
NVCC = nvcc

SRC = vector.cpp barrier.cpp task.cpp manager.cpp
NVSRC = vector.cu

CC_FLAGS = -g -O3 -I/cm/shared/apps/cuda10.0/toolkit/10.0.130/include/ -L/cm/shared/apps/cuda10.0/toolkit/10.0.130/lib64
NVCC_FLAGS = --default-stream per-thread
LL_FLAGS = -fopenmp -lcudart

EXE = vector.out

debug: $(SRC)
	$(CC) $(CC_FLAGS) -c $(SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(NVSRC) -o cuda.o
	$(CC) *.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o $(EXE) $(LL_FLAGS)


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *.o *.out
