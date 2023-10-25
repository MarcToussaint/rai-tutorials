
ipynb_paths =  $(shell find . -maxdepth 1 -type f -name '*.ipynb' -not -name '*checkpoint*' -not -name '*nbconvert*' -not -name '*real*' -printf "%f ")

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
	cp rai.cfg rai.cfg.save
	echo "noInteractivity: true" >>rai.cfg
	echo "bot/blockRealRobot: true" >>rai.cfg
	+@-echo "=========== run $< ======="
	+@-jupyter-nbconvert --to notebook --execute $<
	+@-echo "=========== done $< ======="
	rm -f *.nbconvert.ipynb
	git checkout rai.cfg
