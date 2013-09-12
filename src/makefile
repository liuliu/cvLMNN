INC=-I
LIB=-L
OPENCV=/opt/opencv
LINK1=-L"$(OPENCV)/lib"
LINK2=-lcxcore -lcv -lml -lhighgui
CXXFLAGS = $(INC)"$(OPENCV)/include/opencv/"

all: lmnntest

lmnntest: lmnntest.o mllmnn.o $(OPENCV)/src/ml/ml_inner_functions.o
	g++ $(CXXFLAGS) $(LINK1) -o $@ $^ $(LINK2) -O3

.c.o:
	g++ $< -o $@ -g -c -O3

clean:
	rm *.o

