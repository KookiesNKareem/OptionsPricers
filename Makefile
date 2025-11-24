CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -ffast-math
TARGET = pricer
INCLUDES = -I/opt/homebrew/include -I/opt/homebrew/include/eigen3
LDFLAGS = -L/opt/homebrew/lib
LIBS = -lsleef

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.cpp $(LDFLAGS) $(LIBS) -o $(TARGET) -fno-tree-vectorize

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)