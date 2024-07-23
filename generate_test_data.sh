#!/bin/bash
set -eu

#J=1
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 0.9 --nsteps 7 --type ed --J 1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mbr --J 1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type peps --J 1
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mps --J 1

python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type ed --J 1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mbr --J 1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type peps --J 1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mps --J 1

#J=-1
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 0.9 --nsteps 7 --type ed --J -1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mbr --J -1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type peps --J -1
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mps --J -1

python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type ed --J -1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mbr --J -1
#python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type peps --J -1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 0.9 --nsteps 7 --type mps --J -1
