
ipynb_paths =  $(shell find . -maxdepth 1 -type f -name '*.ipynb' -not -name '*checkpoint*' -not -name '*nbconvert*' -printf "%f ")

#run:
#	@for p in $(ipynb_paths); do jupyter-nbconvert --to notebook --execute $$p; done

run: $(ipynb_paths:%=%.run)
clean: $(ipynb_paths:%=%.clean)
convert: $(ipynb_paths:%=%.convert)

%.ipynb.clean: %.ipynb
	+@-jupyter-nbconvert --clear-output --inplace $<
	rm -f *.nbconvert.ipynb

%.ipynb.convert: %.ipynb
	+@-jupyter-nbconvert --to python $<

%.ipynb.run: %.ipynb
	+@-echo "=========== run $< ======="
	+@-jupyter-nbconvert --to notebook --execute $<
	rm -f *.nbconvert.ipynb
