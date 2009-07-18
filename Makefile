
clean: pyc-clean 

pyc-clean:
	@find $(HERE) -name '*.py[co]' -print0 | xargs -0 -r rm

