#!/bin/bash
for pra in 0 1 2 3 4
do
    python3 trainHCAS.py $pra 0 0 &
    python3 trainHCAS.py $pra 5 0 &
    python3 trainHCAS.py $pra 10 1 &
    python3 trainHCAS.py $pra 15 2 &
    python3 trainHCAS.py $pra 20 5 &
    python3 trainHCAS.py $pra 30 3 &
    python3 trainHCAS.py $pra 40 4 &
    python3 trainHCAS.py $pra 60 4 &
    wait
done
