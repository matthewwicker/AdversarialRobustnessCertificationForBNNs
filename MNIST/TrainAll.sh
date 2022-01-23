#parser.add_argument("--eps", default=0.0)
#parser.add_argument("--lam", default=1.0)
#parser.add_argument("--rob", default=0)
#parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
#parser.add_argument("--opt")
#parser.add_argument("--width", default=24)
#parser.add_argument("--depth", default=1)
for IMG in {0..100}
do
	for WIDTH in 32 64 128 256 512
	do
		for DEPTH in 1 2 3
		do
			for LAMBDA in 0.0 0.25 0.5 0.75
			do
				python3 train.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $IMG &
				python3 train.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt BBB --imnum $IMG &
				python3 train.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				python3 train.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 5 --opt VOGN --imnum $IMG &
				python3 train.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $IMG &
				python3 train.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt NA --imnum $IMG &
				python3 train.py --eps 0.10 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $IMG &
				python3 train.py --eps 0.05 --width $WIDTH --depth $DEPTH --lam $LAMBDA --rob 1 --opt SWAG --imnum $IMG &
				wait
			done
		done
	done
done
