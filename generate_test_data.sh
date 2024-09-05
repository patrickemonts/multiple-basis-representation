#!/bin/bash
set -eu

#J=1, N=2
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type ed --J 1
# python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J 1 --mbr-degree 1
# python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J 1 --mbr-degree 2
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type peps --J 1 --peps-max-chi 3
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mps --J 1 --mps-max-chi 100

#J=1, N=3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type ed --J 1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J 1 --mbr-degree 1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J 1 --mbr-degree 2
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J 1 --mbr-degree 3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type peps --J 1 --peps-max-chi 3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mps --J 1 --mps-max-chi 100

#J=-1, N=2
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type ed --J -1
# python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J -1 --mbr-degree 1
# python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J -1 --mbr-degree 2
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type peps --J -1 --peps-max-chi 3
python manager.py --nx 2 --ny 2 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mps --J -1 --mps-max-chi 100

#J=-1, N=3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type ed --J -1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J -1 --mbr-degree 1
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J -1 --mbr-degree 2
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mbr --J -1 --mbr-degree 3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type peps --J -1 --peps-max-chi 3
python manager.py --nx 3 --ny 3 --hmin 0.3 --hmax 4.9 --nsteps 10 --type mps --J -1 --mps-max-chi 100
