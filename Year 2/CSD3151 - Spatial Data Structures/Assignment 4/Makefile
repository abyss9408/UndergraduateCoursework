########################################################################
# Makefile for Linux
#

ifeq ($(v), std0)
    VFLAG :=  -DTC=12 -DDEFMAIN=0
    ODIR := stdobjs0
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -USOLUTION  -UGEOMLIB_SOLUTION
    pkgName = CS350-framework
else ifeq ($(v), std1)
    VFLAG :=  -DTC=12 -DDEFMAIN=1
    ODIR := stdobjs1
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -USOLUTION  -UGEOMLIB_SOLUTION
    pkgName = CS350-framework
else ifeq ($(v), std2)
    VFLAG :=  -DTC=12 -DDEFMAIN=2
    ODIR := stdobjs2
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -USOLUTION  -UGEOMLIB_SOLUTION
    pkgName = CS350-framework
else ifeq ($(v), sol0)
    VFLAG := -DSOLUTION -DGEOMLIB_SOLUTION -DTC=12 -DDEFMAIN=0
    ODIR := solobjs0
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -DSOLUTION -DGEOMLIB_SOLUTION
    pkgName = CS350-sol
else ifeq ($(v), sol1)
    VFLAG := -DSOLUTION -DGEOMLIB_SOLUTION -DTC=12 -DDEFMAIN=1
    ODIR := solobjs1
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -DSOLUTION -DGEOMLIB_SOLUTION
    pkgName = CS350-sol
else ifeq ($(v), sol2)
    VFLAG := -DSOLUTION -DGEOMLIB_SOLUTION -DTC=12 -DDEFMAIN=2
    ODIR := solobjs2
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -DSOLUTION -DGEOMLIB_SOLUTION
    pkgName = CS350-sol
else
    VFLAG := -DSOLUTION -DGEOMLIB_SOLUTION -DTC=12 -DDEFMAIN=1
    ODIR := solobjs1
    shaders = lightingPhong.vert lightingPhong.frag rt.vert rt.frag
    udflags = -DSOLUTION -DGEOMLIB_SOLUTION
    pkgName = CS350-sol
endif

ifdef c
    pkgName=$(c)-framework
else
endif

# ifndef v
# $(error This makefile needs a flag: .e.g v=std1, or v=sol1)
# endif

# Search for the libs directory
ifneq (,$(wildcard libs))
    LIBDIR := libs
else
    ifneq (,$(wildcard ../libs))
        LIBDIR := ../libs
    else
        ifneq (,$(wildcard ../../libs))
            LIBDIR := ../../libs
        else
            LIBDIR := ../../../libs
        endif
    endif
endif

# Where the compiler will search for source files.
VPATH = $(LIBDIR)/imgui-master $(LIBDIR)/imgui-master/backends

# Where the .o files go
vpath %.o  $(ODIR)

CXX = g++
CFLAGS = -g $(VFLAG) -I. -I$(LIBDIR)/glm -I$(LIBDIR)/imgui-master -I$(LIBDIR)/imgui-master/backends -I$(LIBDIR)  -I$(LIBDIR)/glfw/include -DIMGUI_IMPL_OPENGL_LOADER_GLBINDING2

CXXFLAGS = -std=c++17 $(CFLAGS) -Wno-deprecated

LIBS =  -L/usr/lib/x86_64-linux-gnu -L../$(LIBDIR) -L/usr/lib -L/usr/local/lib -lglbinding -lX11 -lGLU -lGL `pkg-config --static --libs glfw3`

CPPsrc = framework.cpp interact.cpp transform.cpp scene.cpp texture.cpp shapes.cpp object.cpp shader.cpp simplexnoise.cpp fbo.cpp geomlib-main.cpp geomlib-advanced.cpp student_code.cpp
IMGUIsrc = imgui.cpp imgui_widgets.cpp imgui_draw.cpp imgui_demo.cpp imgui_impl_glfw.cpp imgui_impl_opengl3.cpp

headers = framework.h interact.h texture.h shapes.h object.h scene.h shader.h transform.h simplexnoise.h fbo.h geomlib.h student_code.h
srcFiles = $(CPPsrc) $(shaders) $(headers)
extraFiles = framework.vcxproj Makefile

pkgDir = /home/gherron/packages
objs = $(patsubst %.cpp,%.o,$(CPPsrc)) $(patsubst %.cpp,%.o,$(IMGUIsrc))
target = $(ODIR)/framework.exe

$(target): $(objs)
	@echo Link $(target)
	cd $(ODIR) && $(CXX) -g  -o ../$@  $(objs) $(LIBS)

print:
	@echo  VFLAG=$(VFLAG)

help:
	@echo "Try:"
	@echo "    make v=std0 run "
	@echo "    make v=std1 run "
	@echo "    make v=std2 run "
	@echo "    make v=sol0 run "
	@echo "    make v=sol1 run "
	@echo "    make v=sol2 run "

run: $(target)
	LD_LIBRARY_PATH="$(LIBDIR);$(LD_LIBRARY_PATH)" ./$(target)

gdb: $(target)
	LD_LIBRARY_PATH="$(LIBDIR);$(LD_LIBRARY_PATH)" gdb ./$(target)

what:
	@echo VPATH = $(VPATH)
	@echo LIBS = $(LIBDIR)
	@echo CFLAGS = $(CFLAGS)
	@echo VFLAG = $(VFLAG)
	@echo objs = $(objs)
	@echo shaders = $(shaders)
	@echo pkgName = $(pkgName)
	@echo pkgDir = $(pkgDir)
	@echo srcFiles = $(srcFiles)
	@echo extraFiles = $(extraFiles)
	@echo udflags = $(udflags)

clean:
	rm -rf */*.o  dependencies

%.o: %.cpp
	@echo Compile $<  $(VFLAG)
	@mkdir -p $(ODIR)
	@$(CXX) -c $(CXXFLAGS) $< -o $(ODIR)/$@

%.o: %.c
	@echo Compile $< $(VFLAG)
	@mkdir -p $(ODIR)
	@$(CC) -c $(CFLAGS) $< -o $(ODIR)/$@

zip:
	rm -rf $(pkgDir)/$(pkgName) $(pkgDir)/$(pkgName).zip
	mkdir $(pkgDir)/$(pkgName)
	cp $(srcFiles) $(pkgDir)/$(pkgName)
	cp -r $(extraFiles) $(pkgDir)/$(pkgName)
	cp -r ../libs $(pkgDir)/$(pkgName)
	rm -rf $(pkgDir)/$(pkgName)/libs/.hg $(pkgDir)/$(pkgName)/libs/assimp $(pkgDir)/$(pkgName)/libs/bvh
	cd $(pkgDir)/$(pkgName);  unifdef $(udflags)  -m  $(srcFiles) || /bin/true

	cd $(pkgDir);  zip -r $(pkgName).zip $(pkgName)

ws:
	unix2dos $(srcFiles)
	@echo
	@echo ========= TABS:
	@grep -P '\t' $(srcFiles)

dependencies: 
	g++ -MM $(CXXFLAGS) $(CPPsrc) > dependencies

include dependencies
