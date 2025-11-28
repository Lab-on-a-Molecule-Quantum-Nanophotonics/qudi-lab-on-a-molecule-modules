# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
GH_PAGES      = gh_page

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

upload: html
	$(eval TMP := $(shell mktemp -d))
	git worktree add $(TMP) $(GH_PAGES)
	cp -r $(BUILDDIR)/html/* $(TMP)
	git -C $(TMP) add $(TMP)
	git -C $(TMP) commit -a -m "Automatic website build"
	git -C $(TMP) push
	git worktree remove $(TMP)
	git worktree prune
	git push $(GH_PAGES)
	rm -rf $(TMP)

.PHONY: help Makefile upload

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
