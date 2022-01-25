#!/bin/bash
for phi in 0 1 2 3 4
do
    for pra in 0 1 2 3 4
    do
        python3 gridVeriHCAS.py $pra 20 $phi 0 5 &
        python3 gridVeriHCAS.py $pra 20 $phi 1 5 &
        python3 gridVeriHCAS.py $pra 20 $phi 2 5 &
        python3 gridVeriHCAS.py $pra 20 $phi 3 5 &
        python3 gridVeriHCAS.py $pra 20 $phi 4 5 &
        wait
    done
done
