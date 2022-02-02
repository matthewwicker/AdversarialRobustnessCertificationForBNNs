#!/bin/bash
for phi in 0 1 2 3 4
do
    for pra in 0 1 2 3 4
    do
        python3 verifyUpperHCAS.py $pra 0 $phi &
        python3 verifyUpperHCAS.py $pra 5 $phi &
        python3 verifyUpperHCAS.py $pra 10 $phi &
        python3 verifyUpperHCAS.py $pra 15 $phi &
        python3 verifyUpperHCAS.py $pra 20 $phi &
        python3 verifyUpperHCAS.py $pra 30 $phi &
        python3 verifyUpperHCAS.py $pra 40 $phi &
        python3 verifyUpperHCAS.py $pra 60 $phi &
    done
    wait
done
