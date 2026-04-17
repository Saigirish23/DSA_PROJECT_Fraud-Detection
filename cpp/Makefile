# Makefile for graph_algorithms
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall

TARGET = graph_algorithms
SRC    = graph_algorithms.cpp

all: $(TARGET)

$(TARGET): $(SRC) graph_algorithms.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
