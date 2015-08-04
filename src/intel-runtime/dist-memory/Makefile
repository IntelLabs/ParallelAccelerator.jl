
CURRENT_DIR = $(CURDIR)
INCLUDE_DIR =$(CURRENT_DIR)/../include
CC = mpiicpc # C compiler
CXX = mpiicpc
CFLAGS=-I$(INCLUDE_DIR) -I. -DNDEBUG -Wall -Wextra -O0 -mkl -qopenmp -fpic -qoffload-attribute-target=mic -restrict -std=c99 -DMKL_ILP64 -Wno-deprecated -fpermissive
CPPFLAGS=-I$(INCLUDE_DIR)   -I. -Wall -Wextra -DNDEBUG -DSTATISTICS -O0 -mkl -qopenmp -fpic -qoffload-attribute-target=mic -restrict -DMKL_ILP64 -Wno-deprecated -fpermissive
#CPPFLAGS=-I$(INCLUDE_DIR)   -I. -O0 -g 
#CPPFLAGS=-I$(INCLUDE_DIR)  -I. -Wall -Wextra -DSTATISTICS -O2 -g -mkl -qopenmp -fpic -qoffload-attribute-target=mic -restrict -DMKL_ILP64 -Wno-deprecated -fpermissive
LDFLAGS = -O3 -shared   -pthread # linking flags
#LDFLAGS = -g -O0 # linking flags
RM = rm -f  # rm command
TARGET_LIB = libintel-runtime.so # target lib
#TARGET_LIB = test # target lib

OBJECT_DIR=obj
LIB_DIR=$(CURRENT_DIR)/../lib
 
SRCS = Block Point DArray DataMap DataRequest IterMap util Task Interface 


_OBJS = $(SRCS:%=%.o)
OBJS = $(patsubst %,$(OBJECT_DIR)/%,$(_OBJS)) #test.o

HEADERS = $(INCLUDE_DIR)/pse-runtime.h Point.h 
 
.PHONY: all object_dir 

all: object_dir ${TARGET_LIB}

object_dir:
	mkdir -p $(OBJECT_DIR)
lib_dir:
	mkdir -p $(LIB_DIR)

$(TARGET_LIB): $(OBJS) 
	$(CC) ${LDFLAGS} -o $(LIB_DIR)/$@ $^ 

$(OBJECT_DIR)/%.o: %.c $(HEADERS) 
	$(CC) $(CFLAGS) -c $< -o $@
 
$(OBJECT_DIR)/%.o: %.cpp $(HEADERS) %.h
	$(CXX) $(CPPFLAGS) -c $< -o $@

 
.PHONY: clean 

test.o: test.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

#test: test.o $(TARGET_LIB)
#	$(CXX) ${LDFLAGS} -o $@ test.o $(LIB_DIR)/$(TARGET_LIB)
	

clean:
	-${RM} ${OBJS} 

