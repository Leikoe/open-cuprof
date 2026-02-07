NVCC = nvcc
NVCCFLAGS = -arch=sm_89 -I. -lcupti

all: dgemm_example multi_warp_example nested_events_example flash_attention_example

dgemm_example: examples/dgemm_example.cu profiler.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

multi_warp_example: examples/multi_warp_example.cu profiler.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

nested_events_example: examples/nested_events_example.cu profiler.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

flash_attention_example: examples/flash_attention_example.cu profiler.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f dgemm_example multi_warp_example nested_events_example flash_attention_example *.o *.json

.PHONY: all clean
