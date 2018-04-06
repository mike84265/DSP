.PHONY: all clean

CFLAGS+=-std=c++11 -Wall -O3 -DDEBUG
LDFLAGS+=-lm     # link to math library

TARGET=test_hmm train_hmm.o train

all: $(TARGET)
# type make/make all to compile test_hmm

train: train_hmm.h train_hmm.cpp train.cpp
	g++ train_hmm.cpp -c -o train_hmm.o
	g++ train.cpp train_hmm.o -o train

clean:
	$(RM) $(TARGET)   # type make clean to remove the compiled file
