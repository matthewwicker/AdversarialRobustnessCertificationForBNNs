#!/bin/bash
for phi in 1 2 3 4 5 6 7 8 9
do
        python3 gridVeriVCAS.py 1 $phi 0 5 &
        python3 gridVeriVCAS.py 1 $phi 1 5 &
        python3 gridVeriVCAS.py 1 $phi 2 5 &
        python3 gridVeriVCAS.py 1 $phi 3 5 &
        python3 gridVeriVCAS.py 1 $phi 4 5 &
	wait
#    for pra in 1 2 3 4 5 6 7 8 9
#    do
#        python3 gridVeriVCAS.py $pra $phi 0 5 &
#        python3 gridVeriVCAS.py $pra $phi 1 5 &
#        python3 gridVeriVCAS.py $pra $phi 2 5 &
#        python3 gridVeriVCAS.py $pra $phi 3 5 &
#        python3 gridVeriVCAS.py $pra $phi 4 5 &
#        wait
#    done
done
