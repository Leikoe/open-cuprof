NVCC = nvcc
NVCCFLAGS = -arch=sm_89 -I.

all: dgemm_example

dgemm_example: examples/dgemm_example.cu profiler.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f dgemm_example *.o *.json

.PHONY: all clean
