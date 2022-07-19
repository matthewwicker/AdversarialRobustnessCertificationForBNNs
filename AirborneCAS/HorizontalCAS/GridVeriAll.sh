#!/bin/bash
for phi in 0 1 2 3 4
do
    for pra in 0 #1 2 3 4
    do
        python3 gridVeriHCAS.py $pra 0 $phi 0 5 0 &
        python3 gridVeriHCAS.py $pra 0 $phi 1 5 0 &
        python3 gridVeriHCAS.py $pra 0 $phi 2 5 0 &
        python3 gridVeriHCAS.py $pra 0 $phi 3 5 0 &
        python3 gridVeriHCAS.py $pra 0 $phi 4 5 0 &
        #python3 gridUpperHCAS.py $pra 20 $phi 0 5 0 &
        #python3 gridUpperHCAS.py $pra 20 $phi 1 5 0 &
        #python3 gridUpperHCAS.py $pra 20 $phi 2 5 0 &
        #python3 gridUpperHCAS.py $pra 20 $phi 3 5 0 &
        #python3 gridUpperHCAS.py $pra 20 $phi 4 5 0 &

        #python3 gridVeriHCAS.py $pra 20 $phi 0 5 1 &
        #python3 gridVeriHCAS.py $pra 20 $phi 1 5 1 &
        #python3 gridVeriHCAS.py $pra 20 $phi 2 5 1 &
        #python3 gridVeriHCAS.py $pra 20 $phi 3 5 1 &
        #python3 gridVeriHCAS.py $pra 20 $phi 4 5 1 &
        #python3 gridUpperHCAS.py $pra 20 $phi 0 5 1 &
        #python3 gridUpperHCAS.py $pra 20 $phi 1 5 1 &
        #python3 gridUpperHCAS.py $pra 20 $phi 2 5 1 &
        #python3 gridUpperHCAS.py $pra 20 $phi 3 5 1 &
        #python3 gridUpperHCAS.py $pra 20 $phi 4 5 1 &
        wait
    done
done
