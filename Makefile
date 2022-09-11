
.PHONY: all lib clean cleanall distclean
.SUFFIXES: .exe

MAKE_INCLUDE_DIR=..
-include ${MAKE_INCLUDE_DIR}/make.inc


SRC = solvepol.cpp
EXE = $(subst .cpp,.exe,$(SRC))

all: ${EXE}

%.exe: %.o
	 ${CXX} ${CXXFLAGS}  ${CPPFLAGS} -o $@ $< ${SDPA_LIBS}
.cpp.o:
	${CXX} -c ${CXXFLAGS} ${CPPFLAGS} \
		-I${SDPA_DIR}/include ${MUMPS_INCLUDE} ${PTHREAD_INCLUDE} \
	        -o $@ $<
clean:
	rm -f *.o *~
cleanall: clean
	rm -f *.exe
distclean: cleanall

install: all
uninstall:
