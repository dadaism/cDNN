CC=g++
OBJDIR=../lib

all: 
	make libcdnn.so
	mv *.o *.so ${OBJDIR}

cdnn.o: cdnn.c
	${CC} -fPIC -c -Wall $^ -o $@

tensor.o: cdnn_tensor.c
	${CC} -g -fPIC -c -Wall $^ -o $@

convolution.o: cdnn_convolution.c
	${CC} -fPIC -c -Wall $^ -o $@

pooling.o: cdnn_pooling.c
	${CC} -fPIC -c -Wall $^ -o $@

activation.o: cdnn_activation.c
	${CC} -fPIC -c -Wall $^ -o $@

util.o: cdnn_util.c
	${CC} -fPIC -c -Wall $^ -o $@

libcdnn.so: cdnn.o tensor.o convolution.o pooling.o activation.o util.o
	${CC} --shared -fPIC -Wl,-soname,libcdnn.so -o $@ $^ -lc

clean:
	rm ${OBJDIR}/*.o
	rm ${OBJDIR}/*.so
