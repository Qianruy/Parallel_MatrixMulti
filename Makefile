CXX = mpicxx
CXXFLAGS = -std=c++17 -g

ifdef DEBUG
CXXFLAGS += -DDEBUG
endif

pjacobi: pjacobi.cpp
	$(CXX) $(CXXFLAGS) pjacobi.cpp -o pjacobi

clean:
	rm -f pjacobi




