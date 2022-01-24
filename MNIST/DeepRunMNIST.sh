for IMG in {1000..1100}
do
	for WIDTH in 32
	do
		for DEPTH in 1 2 3
		do
			for LAMBDA in 0.0
			do
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $IMG &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $IMG &
				python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				python3 lower.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				python3 lower.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $IMG &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $IMG &
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $IMG &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $IMG &
				I=$(( IMG+101  ))
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $I &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $I &
				python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $I &
				python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $I &
				python3 lower.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $I &
				python3 lower.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $I &
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $I &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $I &
				#python3 upper.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $I &
				#python3 upper.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $I &
			done
			wait
		done
	done
done
