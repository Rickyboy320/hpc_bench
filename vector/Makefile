CC = mpic++
NVCC = nvcc

SRC = vector.cpp variants.cpp
NVSRC = vector.cu

CC_FLAGS = -O3 -I/cm/shared/apps/cuda10.0/toolkit/10.0.130/include/ -L/cm/shared/apps/cuda10.0/toolkit/10.0.130/lib64
NVCC_FLAGS =
LL_FLAGS = -fopenmp -lcudart

EXE = vector

debug: $(SRC)
	$(CC) $(CC_FLAGS) -c $(SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(NVSRC) -o cuda.o
	$(CC) *.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o $(EXE) $(LL_FLAGS)


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result* main.o
