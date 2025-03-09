CXX = clang++
CXXFLAGS = -I/opt/homebrew/Cellar/open-mpi/5.0.7/include -std=c++20
LDFLAGS = -L/opt/homebrew/Cellar/open-mpi/5.0.7/lib -lmpi

TARGET = sobel
OBJS = sobel.o lodepng.o

NUMBER_OF_PROCESSES = 2

INPUT_FOLDER=
OUTPUT_FOLDER=

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

sobel.o: sobel.cpp lodepng.h
	$(CXX) $(CXXFLAGS) -c sobel.cpp

lodepng.o: lodepng.cpp lodepng.h
	$(CXX) $(CXXFLAGS) -c lodepng.cpp

clean:
	rm -f $(TARGET) $(OBJS)

example: $(TARGET)
	mpirun -np $(NUMBER_OF_PROCESSES) ./$(TARGET) example_data example_output

run: $(TARGET)
	mpirun -np $(NUMBER_OF_PROCESSES) ./$(TARGET) $(INPUT_FOLDER) $(OUTPUT_FOLDER)