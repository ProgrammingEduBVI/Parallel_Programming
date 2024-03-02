CC = nvcc
C_FLAGS =
LD_FLAGS =
LIBS =
SOURCES = hello_world.cu vector_add.cu static_shared_memory.cu dynamic_shared_memory.cu
OBJECTS = $(SOURCES:%.cu=%.o)
BINS = $(OBJECTS:%.o=%.bin)
DEPENDS =

all: $(SOURCES) $(OBJECTS) $(BINS)

%.o: %.cu $(DEPENDS)
	$(CC) -c -o $@ $< $(C_FLAGS)

%.bin : %.o
	$(CC) $(C_FLAGS) $(LD_FLAGS) -o $@ $^ $(LIBS)

clean:
	rm $(BINS) $(OBJECTS)                       
