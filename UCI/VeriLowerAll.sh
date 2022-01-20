# Author: Matthew Wicker
for im in {0..250}
do
    python3 lower.py --dataset kin8nm1 --imnum $im &
    python3 lower.py --dataset concrete1 --imnum $im &
    python3 lower.py --dataset wine1 --imnum $im &
    python3 lower.py --dataset powerplant1 --imnum $im &
    python3 lower.py --dataset naval1 --imnum $im &
    python3 lower.py --dataset energy1 --imnum $im &
    python3 lower.py --dataset yacht1 --imnum $im &
    wait
done
