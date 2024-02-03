echo "noInteractivity: true" >>rai.cfg
echo "bot/blockRealRobot: true" >>rai.cfg
echo "=========== run $1 ======="
jupyter-nbconvert --execute --inplace $1
echo "=========== done $1 ======="
rm -f *.nbconvert.ipynb
git checkout rai.cfg
