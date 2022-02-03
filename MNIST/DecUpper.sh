#parser.add_argument("--imnum", default=0)
#parser.add_argument("--eps", default=0.05)
#parser.add_argument("--lam", default=0.0)
#parser.add_argument("--rob", default=5)
#parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
#parser.add_argument("--opt", default="VOGN")
#parser.add_argument("--width", default=32)
#parser.add_argument("--depth", default=1)
for imnum in {0..100}
do
	python3 decision_upper.py --imnum $imnum --depth 1  --data "mnist" &
	python3 decision_upper.py --imnum $imnum --depth 2  --data "mnist" &
	python3 decision_upper.py --imnum $imnum --depth 3  --data "mnist" &
	python3 decision_upper.py --imnum $imnum --depth 1  --data "fmnist" &
	python3 decision_upper.py --imnum $imnum --depth 2  --data "fmnist" &
	python3 decision_upper.py --imnum $imnum --depth 3  --data "fmnist" &
        wait
done
