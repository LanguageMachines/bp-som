#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([bp-som],[0.1],[lamasoftware@science.ru.nl])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([.])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

AX_REQUIRE_DEFINED([AX_CXX_COMPILE_STDCXX_17])

# Checks for programs.
AC_PROG_CXX( [g++] )
AX_CXX_COMPILE_STDCXX_17

# when running tests, use CXX
AC_LANG([C++])

# use libtool
LT_INIT

#checks for libraries.

# Checks for header files.


# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_CHECK_FUNCS([floor pow rint sqrt ])

AC_CONFIG_FILES([
  Makefile
  m4/Makefile
  src/Makefile
  include/Makefile
  include/bpsom/Makefile
  examples/Makefile
])
AC_OUTPUT
