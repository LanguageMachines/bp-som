#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([bp-som],[0.1],[lamasoftware@science.ru.nl])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([.])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADER([config.h])

if test x"${CXXFLAGS+set}" = xset; then
  # the user set CXXFLAGS; don't override it.
  cxx_flags_were_set=true
else
  cxx_flags_were_set=false
fi
if $cxx_flags_were_set; then
  CXXFLAGS=$CXXFLAGS
fi

# Checks for programs.
AC_PROG_CXX( [g++] )

# when running tests, use CXX
AC_LANG([C++])

# use libtool
AC_PROG_LIBTOOL
LT_INIT

#checks for libraries.

# Checks for header files.


# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_CONFIG_FILES([
  Makefile
  m4/Makefile
  src/Makefile
  include/Makefile
  include/bpsom/Makefile
  examples/Makefile
])
AC_OUTPUT
