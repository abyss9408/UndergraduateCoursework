ECHO OFF
:: Batch file: csd2101-project.bat
:: Author: Parminder Singh
:: Date: July 16, 2024
:: Revision: 1.0

::     .d8888b.   .d8888b.  8888888b.          .d8888b.   d888   .d8888b.   .d88     ::
::    d88P  Y88b d88P  Y88b 888  "Y88b        d88P  Y88b d8888  d88P  Y88b d8888     ::
::    888    888 Y88b.      888    888               888   888  888    888   888     ::
::    888         "Y888b.   888    888             .d88P   888  888    888   888     ::
::    888            "Y88b. 888    888         .od888P"    888  888    888   888     ::
::    888    888       "888 888    888 888888 d88P"        888  888    888   888     ::
::    Y88b  d88P Y88b  d88P 888  .d88P        888"         888  Y88b  d88P   888     ::
::     "Y8888P"   "Y8888P"  8888888P"         888888888  8888888 "Y8888P"  8888888   ::
                                                                                

REM Set the source directory to "projects".
SET "projects-dir=projects"

REM Set the library directory.
SET "lib-dir=lib"

REM Set start code directory for csd-2101.
SET "csd-2100-starter-kit=%lib-dir%\csd-2101"

REM Setting Git directory path
SET "GitDir=%~dp0.git"

REM Setting build directory path
SET "BuildDir=%~dp0build"

REM Setting Git modules path
SET "GitModules=%~dp0.gitmodules"

REM Setting CMake file path
SET "CMakeFile=%~dp0CMakeLists.txt"

CLS
:MENU
	ECHO.
	ECHO .....................................................
	ECHO Select your task, or Esc to EXIT.
	ECHO .....................................................
	ECHO   S   - Setup Project Environment
	ECHO   D   - Delete Project Environment
	ECHO   B   - Build Project
	ECHO   C   - Clean and Build Project
	ECHO   R   - Refresh Project
	ECHO   E   - EXIT
	ECHO   0   - Create Project
	ECHO.
	SET /P M=Type option then press ENTER:

	IF /I %M%==S GOTO CMAKE_BUILD
	IF /I %M%==D GOTO CLEAN
	IF /I %M%==B GOTO BUILD_ALL
	IF /I %M%==C GOTO CLEAN_BUILD_ALL
	IF /I %M%==R GOTO BUILDCMAKELISTS
	IF /I %M%==E GOTO :EOF
	IF %M%==0 (
		GOTO CREATE_PROJECT_FINAL
	) ELSE (
	    ECHO Invalid choice. Please enter valid option from menu.
	    PAUSE
	    GOTO MENU
	)

:CREATE_PROJECT_FINAL
	CALL :CHECK_OPENGL_LIBS "%lib-dir%"
	CALL :COPY_FROM_REMOTE_URL https://giraphics.github.io/csd-2101/csd2101+proj.zip projects

	set "dpml-src-path=%projects-dir%/csd2101+proj/gfx"
    CALL :COPY_RESOURCE %dpml-src-path% %lib-dir%/gfx

	set "content-src-path=%projects-dir%/csd2101+proj/content"
    CALL :COPY_RESOURCE %content-src-path% content

	GOTO BUILDCMAKELISTS

:COPY_FROM_REMOTE_URL
	setlocal
		SET "remote_url=%~1"
		SET "destination_folder=%~2"

		FOR /f "delims=/ tokens=*" %%a in ("%remote_url%") do (
			SET "filename_ext=%%~nxa"
		    SET "filename=%%~na"
		)

		curl -L -o "%filename_ext%" "%remote_url%"
		IF %errorlevel% neq 0 (ECHO Failed to download the file.)

		tar -xf "%filename_ext%" -C .
		IF %errorlevel% neq 0 (ECHO Failed to unzip the file.)
		
		IF EXIST "%filename_ext%" ECHO y|del "%filename_ext%"
		xcopy "%filename%" "%destination_folder%/%filename%" /s /i /-y

		IF EXIST "%filename%" ECHO y|rmdir /s "%filename%"
	endlocal
	goto :eof

:: copy resources
:COPY_RESOURCE
	call :CHECK_OPENGL_LIBS "%lib-dir%"
	set "source-path=%~1"
	set "destination-path=%~2"

	if not exist "%destination-path%" (
		mkdir "%destination-path%"
	)

	if not exist "%source-path%" (
	    echo [WARNING!] %source-path% directory does not exists.
	    pause
	    GOTO MENU
	) else (
		xcopy "%source-path%" "%destination-path%" /s /e /i /-y
		rmdir /s /q "%source-path%"
	)
	goto :eof


:: Check OpenGL dependencies
:CHECK_OPENGL_LIBS
	SET "folder=%~1"
	IF not EXIST "%folder%" (
	    ECHO [WARNING!] OpenGL 4.5 is not configured.
	    ECHO Choose the menu option: Press S to set up OpenGL 4.5.
	    REM You can add further instructions or actions here.
	    PAUSE
	    GOTO MENU
	)
	goto :eof

:CMAKE_BUILD
	SET "GitFolder=%~dp0.git"
	SET "GitModules=%~dp0.gitmodules"

	IF EXIST "%GitFolder%" ECHO y|rmdir /s "%GitFolder%"
    CALL git init
    CALL git config core.autocrlf false
	CALL git submodule add https://github.com/g-truc/glm ./lib/glm	

	IF EXIST "%GitFolder%" ECHO y|rmdir /s "%GitFolder%"
	IF EXIST "%GitModules%" ECHO y|del "%GitModules%"

	SET "folders=build %projects-dir%"

	REM Loop through each folder name and create it
	FOR %%i in (%folders%) do (
	    mkdir "%%i"
	)
	GOTO BUILDCMAKELISTS
	GOTO MENU

:BUILD_ALL
	cmake --build build --config Release
	GOTO MENU

:CLEAN_BUILD_ALL
	cmake --build build --clean-first --config Release
	GOTO MENU

:CLEAN
	IF EXIST "%GitDir%" ECHO y|rmdir /s "%GitDir%"
	IF EXIST "%BuildDir%" ECHO y|rmdir /s "%BuildDir%"
	IF EXIST "%lib-dir%" ECHO y|rmdir /s "%lib-dir%"
	IF EXIST "%GitModules%" ECHO y|del "%GitModules%"
	IF EXIST "%CMakeFile%" ECHO y|del "%CMakeFile%"  
	GOTO MENU

:PLAYLOGS
	:: Below Playback log are har
	GOTO MENU

:BUILDCMAKELISTS
	IF EXIST "%CMakeFile%" ECHO y|del "%CMakeFile%"  

	(
		ECHO # Projects Settings
		ECHO.
		ECHO cmake_minimum_required(VERSION 2.8...3.13^)
		ECHO.
		ECHO project (csd2101+proj^)
		ECHO.

		ECHO include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/glm^)
		ECHO.
		
		REM Please gather the target folder first before the project is build otherwise it will capture build folder as well
		REM ECHO set(DisableWarnings "/wd5045" "/wd4365" "/wd4242" "/wd4365" "/wd4820" "/wd4514" "/wd5039" "/wd4668" "/wd5219" "/wd4061" "/wd4091" "/wd4464" "/wd4100" "/wd4458" "/wd4267" "/wd4244" "/wd4305" "/wd4189" "/wd4710" "/wd4711" "/wd4201" "/wd4626" ^)
		ECHO.

		IF EXIST "%projects-dir%" (
			FOR /d %%i in ("%projects-dir%\*") do (
			    REM ECHO %%~nxi

			ECHO file(GLOB_RECURSE %%~nxi_source_files 
			ECHO     ${CMAKE_CURRENT_LIST_DIR}/%projects-dir%/%%~nxi/*.[ch]pp
			ECHO     ${CMAKE_CURRENT_LIST_DIR}/%projects-dir%/%%~nxi/*.h
			ECHO ^)
		    ECHO add_executable(%%~nxi 
		    ECHO     ${%%~nxi_source_files} 
		    ECHO ^)
			ECHO.
		    ECHO target_link_directories(%%~nxi
		    ECHO     PRIVATE
		    ECHO         ^$^<$^<CONFIG:Debug^>:${CMAKE_CURRENT_LIST_DIR}/lib/gfx/lib/Debug^>
		    ECHO         ^$^<$^<CONFIG:Release^>:${CMAKE_CURRENT_LIST_DIR}/lib/gfx/lib/Release^>
		    ECHO ^)
			ECHO target_link_libraries(%%~nxi
			ECHO     GfxLib
			ECHO     gdiplus
			ECHO ^) 
			ECHO.
			ECHO set_property(TARGET %%~nxi PROPERTY CXX_STANDARD 20^) 
			ECHO IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"^)
			ECHO     target_compile_options(%%~nxi PRIVATE -Wall ${DisableWarnings}^)
			ECHO elseif (MSVC^)
			ECHO     target_compile_options(%%~nxi PRIVATE /W3 /WX-^)
			ECHO endif(^)
			ECHO target_include_directories(%%~nxi PRIVATE ${CMAKE_CURRENT_LIST_DIR}/%projects-dir%/%%~nxi/include^)
			ECHO.
			)
		)
	) >> "CMakeLists.txt"

	:: Create CMake project
	SET curDir=%~dp0
	PUSHD %curDir%build
	cmake -G "Visual Studio 17 2022" ..
	POPD
	GOTO MENU
