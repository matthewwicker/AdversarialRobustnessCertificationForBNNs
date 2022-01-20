# Author: Matthew Wicker
for im in {0..250}
do
    python3 upper.py --dataset kin8nm1 --imnum $im &
    python3 upper.py --dataset concrete1 --imnum $im &
    python3 upper.py --dataset wine1 --imnum $im &
    python3 upper.py --dataset powerplant1 --imnum $im &
    python3 upper.py --dataset naval1 --imnum $im &
    python3 upper.py --dataset energy1 --imnum $im &
    python3 upper.py --dataset yacht1 --imnum $im &
    wait
done
