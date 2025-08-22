
if(NOT "/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitinfo.txt" IS_NEWER_THAN "/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/deniz/Documents/split-dev/build/gnuplot-iostream-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/deniz/Documents/split-dev/build/gnuplot-iostream-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout --config "advice.detachedHead=false" "https://github.com/dstahlke/gnuplot-iostream.git" "gnuplot-iostream-src"
    WORKING_DIRECTORY "/home/deniz/Documents/split-dev/build"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/dstahlke/gnuplot-iostream.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout 3a8c001a38304d7bfebb9edf91df5be8e3deaa57 --
  WORKING_DIRECTORY "/home/deniz/Documents/split-dev/build/gnuplot-iostream-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '3a8c001a38304d7bfebb9edf91df5be8e3deaa57'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/deniz/Documents/split-dev/build/gnuplot-iostream-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/deniz/Documents/split-dev/build/gnuplot-iostream-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitinfo.txt"
    "/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/deniz/Documents/split-dev/build/gnuplot-iostream-prefix/src/gnuplot-iostream-stamp/gnuplot-iostream-gitclone-lastrun.txt'")
endif()

