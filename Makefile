CC=clang++
NVCC=nvcc
CFLAGS=-Wall -O3 -std=c++11 -m64
CUDAFLAGS=--ptxas-options=-v --std=c++11 -arch=sm_30 -lineinfo --use_fast_math
CUDAGFLAGS=--ptxas-options=-v -arch=sm_30 -lineinfo -G -g --std=c++11 -Xptxas "-preserve-relocs" -Xnvlink "-preserve-relocs"
ICUDA=-I/usr/local/cuda/include
LCUDA=-L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn

INC=-I /usr/include -I include
LINK=-lgflags

BINDIR=bin
OBJDIR=build
SRCDIR=src
RESDIR=result

_OBJS=ffn.o readubyte.o
OBJS=$(patsubst %,$(OBJDIR)/%,$(_OBJS))

all: $(BINDIR)/test

test:
	@$(BINDIR)/test

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
		$(CC) -c $(CFLAGS) $(INC) $(ICUDA) -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
		$(NVCC) -c $(CUDAFLAGS) $(INC) -o $@ $<

$(BINDIR)/test: $(OBJS) $(OBJDIR)/test.o
	@mkdir -p $(BINDIR)
	@mkdir -p $(RESDIR)
		$(CC) $(LINK) -o $(BINDIR)/test $^ $(LCUDA)

clean:
	@rm -rf $(OBJDIR) $(BINDIR)
