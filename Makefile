
clean: pyc-clean doc-clean

pyc-clean:
	@find $(HERE) -name '*.py[co]' -print0 | xargs -0 -r rm

doc-clean:
	cd doc; make clean

