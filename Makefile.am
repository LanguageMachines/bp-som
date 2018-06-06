ACLOCAL_AMFLAGS =-I m4 --install

SUBDIRS = src include examples m4

EXTRA_DIST = boostrap.sh AUTHORS TODO NEWS README.md LICICEN.md codemeta.json

ChangeLog: NEWS
	git pull; git2cl > ChangeLog
