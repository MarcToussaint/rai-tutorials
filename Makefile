
ipynb_paths =  $(shell find . -maxdepth 1 -type f -name '*.ipynb' -not -name '*checkpoint*' -not -name '*nbconvert*' -printf "%f ")

#run:
#	@for p in $(ipynb_paths); do jupyter-nbconvert --to notebook --execute $$p; done

run: $(ipynb_paths:%=%.run)
clean: $(ipynb_paths:%=%.clean)
nometa: $(ipynb_paths:%=%.nometa)
convert: $(ipynb_paths:%=%.convert)

%.ipynb.clean: %.ipynb
	+@-jupyter-nbconvert --clear-output --inplace $<
	rm -f *.nbconvert.ipynb

%.ipynb.nometa: %.ipynb
	-cat $< | tr '\n' '\f' | sed -e 's/"metadata": {[^\f]*\f[^\f]*execution[^\f]*\f[^\f]*\f[^\f]*\f[^\f]*\f[^\f]*\f[^\f]*\f[^\f]*\f/"metadata": {},\f/g'  | tr '\f' '\n' > $<.z
	mv $<.z $<

%.ipynb.convert: %.ipynb
	+@-jupyter-nbconvert --to python $<

%.ipynb.run: %.ipynb
	echo "noInteractivity: true" >>rai.cfg
	echo "bot/blockRealRobot: true" >>rai.cfg
	+@-echo "=========== run $< ======="
	+@-jupyter-nbconvert --execute --inplace $<
	+@-echo "=========== done $< ======="
	rm -f *.nbconvert.ipynb
	git checkout rai.cfg
