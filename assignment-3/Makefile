matmul: main.c
	@gcc -o $@ $? -l OpenCL

matmul-debug: main.c
	@gcc -DDEBUG -o $@ $? -l OpenCL

archive: main.c main.template matmul.cl Makefile bench.sh README.md
	tar -zcf assignment-3.tar.gz $^
