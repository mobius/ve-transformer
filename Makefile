CC = gcc
NCC = /opt/nec/ve/bin/ncc
CFLAGS = -O2 -std=c11 -Wall -Wextra
SOURCES = src/transformer.c src/demo.c
NLC_HOME = /opt/nec/ve/nlc/3.1.0
NLC_FLAGS = -DVT_USE_NLC -I$(NLC_HOME)/include -fopenmp
FAST_FLAGS = $(NLC_FLAGS) -DVT_NLC_ATTENTION
NLC_LIBS = -L$(NLC_HOME)/lib -lcblas -lblas_openmp -Wl,-rpath,$(NLC_HOME)/lib
NCC_LIB = /opt/nec/ve/ncc/5.4.1/lib
VE_LINK = -Wl,-rpath-link,$(NCC_LIB)
.PHONY: cpu ve nlc fast resident test bench clean
cpu: build/transformer-cpu
ve: build/transformer-ve build/benchmark-ve
build:
	mkdir -p build
build/transformer-cpu: $(SOURCES) src/transformer.h | build
	$(CC) $(CFLAGS) $(SOURCES) -lm -o $@
build/transformer-ve: $(SOURCES) src/transformer.h | build
	$(NCC) -O2 -std=c11 $(SOURCES) $(VE_LINK) -lm -o $@
test: cpu
	python3 tests/check_reference.py build/transformer-cpu
build/benchmark-cpu: src/transformer.c src/benchmark.c src/transformer.h | build
	$(CC) $(CFLAGS) src/transformer.c src/benchmark.c -lm -o $@
build/benchmark-ve: src/transformer.c src/benchmark.c src/transformer.h | build
	$(NCC) -O2 -std=c11 src/transformer.c src/benchmark.c $(VE_LINK) -lm -o $@
nlc: build/transformer-ve-nlc build/benchmark-ve-nlc build/shape-ve-nlc
fast: build/transformer-ve-fast build/benchmark-ve-fast build/shape-ve-fast build/infer-ve-fast
build/transformer-ve-fast: $(SOURCES) src/transformer.h | build
	$(NCC) -O2 -std=c11 $(FAST_FLAGS) $(SOURCES) $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/benchmark-ve-fast: src/transformer.c src/benchmark.c src/transformer.h | build
	$(NCC) -O2 -std=c11 $(FAST_FLAGS) src/transformer.c src/benchmark.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/shape-ve-fast: src/transformer.c tests/shape_fixture.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc $(FAST_FLAGS) src/transformer.c tests/shape_fixture.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/infer-ve-fast: src/transformer.c src/infer.c src/transformer.h | build
	$(NCC) -O2 -std=c11 $(FAST_FLAGS) src/transformer.c src/infer.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/infer-cpu: src/transformer.c src/infer.c src/transformer.h | build
	$(CC) $(CFLAGS) src/transformer.c src/infer.c -lm -o $@
build/infer-ve: src/transformer.c src/infer.c src/transformer.h | build
	$(NCC) -O2 -std=c11 src/transformer.c src/infer.c $(VE_LINK) -lm -o $@
build/infer-ve-nlc: src/transformer.c src/infer.c src/transformer.h | build
	$(NCC) -O2 -std=c11 $(NLC_FLAGS) src/transformer.c src/infer.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/transformer-ve-nlc: $(SOURCES) src/transformer.h | build
	$(NCC) -O2 -std=c11 $(NLC_FLAGS) $(SOURCES) $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/benchmark-ve-nlc: src/transformer.c src/benchmark.c src/transformer.h | build
	$(NCC) -O2 -std=c11 $(NLC_FLAGS) src/transformer.c src/benchmark.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
build/shape-cpu: src/transformer.c tests/shape_fixture.c src/transformer.h | build
	$(CC) $(CFLAGS) -Isrc src/transformer.c tests/shape_fixture.c -lm -o $@
build/shape-ve: src/transformer.c tests/shape_fixture.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc src/transformer.c tests/shape_fixture.c $(VE_LINK) -lm -o $@
build/shape-ve-nlc: src/transformer.c tests/shape_fixture.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc $(NLC_FLAGS) src/transformer.c tests/shape_fixture.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
bench: build/benchmark-cpu
	./build/benchmark-cpu
clean:
	rm -f build/transformer-cpu build/transformer-ve build/benchmark-cpu build/benchmark-ve build/transformer-ve-nlc build/benchmark-ve-nlc build/shape-cpu build/shape-ve build/shape-ve-nlc build/infer-cpu build/infer-ve build/infer-ve-nlc build/transformer-ve-fast build/benchmark-ve-fast build/shape-ve-fast build/infer-ve-fast build/check-workspace-cpu build/check-workspace-ve build/check-workspace-ve-fast build/resident-cpu build/resident-ve-fast

build/check-workspace-cpu: src/transformer.c tests/check_workspace.c src/transformer.h | build
	$(CC) $(CFLAGS) -Isrc src/transformer.c tests/check_workspace.c -lm -o $@
build/check-workspace-ve: src/transformer.c tests/check_workspace.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc src/transformer.c tests/check_workspace.c $(VE_LINK) -lm -o $@
build/check-workspace-ve-fast: src/transformer.c tests/check_workspace.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc $(FAST_FLAGS) src/transformer.c tests/check_workspace.c $(VE_LINK) $(NLC_LIBS) -lm -o $@

resident: build/resident-cpu build/resident-ve-fast
build/resident-cpu: src/transformer.c src/resident.c src/transformer.h | build
	$(CC) $(CFLAGS) -Isrc src/transformer.c src/resident.c -lm -o $@
build/resident-ve-fast: src/transformer.c src/resident.c src/transformer.h | build
	$(NCC) -O2 -std=c11 -Isrc $(FAST_FLAGS) src/transformer.c src/resident.c $(VE_LINK) $(NLC_LIBS) -lm -o $@
