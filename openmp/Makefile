CXX = clang++
CXXFLAGS = -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp -std=c++20
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

TARGET = sobel
OBJS = sobel.o lodepng.o

NUMBER_OF_THREADS = 1
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
	./$(TARGET) example_data example_output $(NUMBER_OF_THREADS)

run: $(TARGET)
	./$(TARGET) $(INPUT_FOLDER) $(OUTPUT_FOLDER) $(NUMBER_OF_THREADS)