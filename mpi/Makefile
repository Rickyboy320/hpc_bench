CC = mpic++

SRC = vector.cpp variants.cpp

CC_FLAGS = -O3
LL_FLAGS = -fopenmp

EXE = vector.out

debug: $(SRC)
	$(CC) $(CC_FLAGS) -c $(SRC)
	$(CC) *.o -o $(EXE) $(LL_FLAGS)


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *.o *.out
