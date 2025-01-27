SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)

BASE := fastCA

HEADERS := $(wildcard *.H)

PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)

PYBIND_SUFFIX := $(shell python3-config --extension-suffix)

PYTHON_LIBRARY := ${BASE}${PYBIND_SUFFIX}

ALL: ${PYTHON_LIBRARY}

# CXXFLAGS := -O3  -Wall -Wextra -shared -std=c++17 -fPIC -fopenmp ${PYBIND_INCLUDES}

CXXFLAGS := -O3  -Wall -Wextra -shared -std=c++17 -fPIC ${PYBIND_INCLUDES}

%.o : %.cpp
	g++ ${CXXFLAGS} -c $<

${PYTHON_LIBRARY}: ${OBJECTS} ${HEADERS}
	g++ ${CXXFLAGS} -o $@ $<


print-%: ; @echo $* is $($*)