SOURCE_TEX=*.tex
SOURCE_GNUPLOT=*.gpi
LOCAL_MAKEFILE=Makefile.local

# TODO:
#  - allow extra dependencies for foobar.pdf in local Makefile (use ::)
#  - extra target make list-images, make list-tex...
#  - using include $(RULES) instead of $(MAKE1), and GNU remaking of
# $(RULES) when necessary.
#  + inclusion of TeX files and packages/classes
#  - auto determine whether TeX / LaTeX / ConTeXt / LuaTeX / ...
#  - factor tex log coloring via a function
#  - allow empty colors
#  - prevent control going back to Makefile to allow calls such as
#    make -f /some/where/this/Makefile

SUBDIR=.make
RULES=$(SUBDIR)/rules.mk
RULES_PLACE=$(RULES)
MAKE1=$(MAKE) --no-print-directory

PDFLATEX=pdflatex
BIBTEX=bibtex
GNUPLOT=gnuplot
# Disable stupid dash echo that does not understand -e
ECHO=/bin/echo -e
SED=/bin/sed -r

# Colors
# To remove a color: replace by ``tput sgr0''
SED_PROTECT=$(SED) -e 's/[[()\]/\\&/g'
COLOR_FILE=tput bold;tput setaf 4
COLOR_WARNING=tput setaf 9
COLOR_FONT=tput setaf 8
COLOR_PACKAGE=tput setaf 3
COLOR_OVERFULL=tput setaf 5

.PHONY: all clean distclean rules rules-force

all: rules
	@$(MAKE1) -f $(RULES) all

clean: rules
	@$(MAKE1) -f $(RULES) clean

distclean: rules
	@$(MAKE1) -f $(RULES) clean
	rm -rf $(SUBDIR)

rules:
	@(if [ -f $(RULES) ] ; then \
	  make --no-print-directory -f $(RULES) rules-comp; \
	else \
	  echo "Creating $(RULES_PLACE)...";\
	  $(MAKE1) rules-force; \
	fi)

rules-force:
	@[ -d $(SUBDIR) ] || mkdir $(SUBDIR)
	@( $(ECHO) "# Auto generated" ;\
	$(ECHO) "\n# -- TeX files --" ;\
	rules_dep='';\
	if [ "$(SOURCE_TEX)" != x$(SOURCE_TEX) ] ; then \
	  rules_dep="$$rules_sep "$$(echo $(SOURCE_TEX)); \
	fi;\
	all_p1=''; all_p2=''; clean='';\
	for s in $(SOURCE_TEX); do \
	  $(ECHO) "\n# -- Rules for $$s --" ;\
	  b=`basename $$s  .tex` ; wb="" ;\
	  clean="$$clean $(SUBDIR)/$$b.clean";\
	  all_p1="$$all_p1 $(SUBDIR)/$$b.p1";\
	  all_p2="$$all_p2 $(SUBDIR)/$$b.p2";\
	  if grep -q '^[^%]*\\bibliography' $$s ; then wb=" $$b.bbl"; fi ;\
	  dep='';\
	  for p in $$($(SED) \
	    -ne \ 's/.*\\usepackage(\[[^]]*\])?\{([^}]*)\}.*/\2.sty/gp' \
	    -ne \ 's/.*\\documentclass(\[[^]]*\])?\{([^}]*)\}.*/\2.cls/gp' \
	    < $$s) ; do [ -f $$p ] && dep="$$dep $$p"; done;\
	  $(ECHO) "\n$(SUBDIR)/$$b.pdf: $(SUBDIR)/$$b.p1 $(SUBDIR)/$$b.p2";\
	  $(ECHO) "\n$(SUBDIR)/$$b.p1: $$b.tex $$dep \
	    $(SUBDIR)/$$b.img $(SUBDIR)/$$b.saux";\
	  $(ECHO) "\t$(PDFLATEX) $$b" ;\
	  $(ECHO) "\t@echo > $(SUBDIR)/$$b.p1";\
	  $(ECHO) "\t@$(SED) -ne \"\
	    /^LaTeX Warning:/ { :a N;\
	    /\\\\n\\\$$\$$/!ba; s/\\\\n\\\$$\$$//;\
	    s/^/$$( ($(COLOR_WARNING))|$(SED_PROTECT))/;\
	    s/\\\$$\$$/$$( (tput sgr0)|$(SED_PROTECT))/;\
	    p; };\
	    /^LaTeX Font Warning:/ { :b N;\
	    /\\\\n\\\$$\$$/!bb; s/\\\\n\\\$$\$$//;\
	    s/^/$$( ($(COLOR_FONT))|$(SED_PROTECT))/;\
	    s/\\\$$\$$/$$( (tput sgr0)|$(SED_PROTECT))/;\
	    p; };\
	    /^(Package|Class) .* Warning:/ { :c N;\
	    /\\\\n\\\$$\$$/!bc; s/\\\\n\\\$$\$$//;\
	    s/^/$$( ($(COLOR_PACKAGE))|$(SED_PROTECT))/;\
	    s/\\\$$\$$/$$( (tput sgr0)|$(SED_PROTECT))/;\
	    p; };\
	    /^Overfull ..box/ { \
	    s/^/$$( ($(COLOR_OVERFULL))|$(SED_PROTECT))/;\
	    s/\\\$$\$$/$$( (tput sgr0)|$(SED_PROTECT))/;\
	    p; };\
	    \" $$b.log > $(SUBDIR)/$$b.p1";\
	  $(ECHO) "\n$(SUBDIR)/$$b.p2: $(SUBDIR)/$$b.p1";\
	  $(ECHO) "\t@if [ -s $(SUBDIR)/$$b.p1 ]; then\\";\
	  $(ECHO) "\t  echo '  $$($(COLOR_FILE))$$b.tex$$(tput sgr0):';\\";\
	  $(ECHO) "\t  cat $(SUBDIR)/$$b.p1; fi; touch $(SUBDIR)/$$b.p2";\
	  $(ECHO) "\n$(SUBDIR)/$$b.saux: $$b.aux $$wb" ;\
	  $(ECHO) "\t@( [ -f $(SUBDIR)/$$b.saux ] \
	    && diff -q $$b.aux $(SUBDIR)/$$b.saux > /dev/null ) \
	    || cp $$b.aux $(SUBDIR)/$$b.saux\n" ;\
	  $(ECHO) "$$b.aux: $$b.tex\n\t$(PDFLATEX) $$b\n" ;\
	  $(ECHO) "$$b.bbl: $$b.aux\n\t$(BIBTEX) $$b\n\t$(PDFLATEX) $$b\n" ;\
	  img_files_in=$$($(SED) -ne '/^ *%/d' -e  \
	    's/.*\\includegraphics(\[[^]]*\]|)\{([^}]*)\}.*/\2/gp'  < $$s \
	    | tr '\n' ' ');\
	  img_files=''; for f in $$img_files_in; do\
	    for e in .png .pdf .jpg .jpeg ; do\
	      if [ -f $$f$$e ] ; then\
	        img_files="$$img_files $$f$$e"; continue 2;\
	      fi;\
	    done;\
	  done;\
	  $(ECHO) "\n$(SUBDIR)/$$b.img: $$img_files";\
	  $(ECHO) "\t@touch $(SUBDIR)/$$b.img";\
	  $(ECHO) "\n$(SUBDIR)/$$b.clean:\n\trm -f $$b.aux $$b.toc $$b.bbl \
	    $$b.pdf $$b.log $$b.blg $$b.vrb $$b.out $$b.nav $$b.snm \
	    $(SUBDIR)/$$b.*\n" ;\
	done ;\
	if [ "x$(SOURCE_GNUPLOT)" != "x$$(echo $(SOURCE_GNUPLOT))" ] ; then \
	rules_dep="$$rules_dep "$$(echo $(SOURCE_GNUPLOT)); \
	  $(ECHO) "\n# -- Gnuplot --" ;\
	for s in $(SOURCE_GNUPLOT); do \
	  out=$$($(SED) -ne "s/set\\s+output\\s+'([^']*)'/\\1/gp" < $$s \
	    | tr '\n' ' ' ); \
	  gnuplot_out="$$gnuplot_out $$out"; \
	  $(ECHO) "\n$$out: $$s\n\t$(GNUPLOT) $$s\n";\
	  clean="$$clean clean-gnuplot";\
	done ;\
	$(ECHO) "\ngnuplot-clean:\n\trm -f $$gnuplot_out";\
	fi;\
	$(ECHO) "\n# -- Global --\n";\
	$(ECHO) "\nall: all-auto\n\nall-auto:$$all_p1 $$all_p2";\
	$(ECHO) "\nclean: clean-auto\n\nclean-auto:$$clean";\
	$(ECHO) "\n.PHONY: all all-auto clean clean-auto";\
	$(ECHO) "\n# -- Recursivity --";\
	$(ECHO) "\nrules-comp:" ;\
	$(ECHO) "\t@\$$(MAKE) -f Makefile RULES_PLACE=$(SUBDIR)/rules.tmp\
	  rules-force";\
	$(ECHO) "\t@diff -q $(SUBDIR)/rules.tmp $(RULES) ||\
	  cp $(SUBDIR)/rules.tmp $(RULES)";\
	$(ECHO) "\t@rm -f $(SUBDIR)/rules.tmp";\
	if [ -f $(LOCAL_MAKEFILE) ] ; then \
	  $(ECHO) "\n# -- Local Makefile included --";\
	  $(ECHO) "# You may use automatic targets all-auto and clean-auto.";\
	  $(ECHO) "\ninclude $(LOCAL_MAKEFILE)";\
	fi;\
	) > $(RULES_PLACE)

#  	$(ECHO) "\n# -- Povray --" ;\
#  	for s in *.pov; do b=$${s%.pov} ;\
#  	  $(ECHO) "$$b.png: $$b.pov\n\t$(POVRAY) -i$$b.pov -o$$b.png\n" ;\
#  	done ;\
#  	$(ECHO) -n "povray-clean:\n\trm -f "; for f in *.pov; do \
#  	  $(ECHO) -n "$${f%pov}png " ;\
#  	done ; $(ECHO) ;\

