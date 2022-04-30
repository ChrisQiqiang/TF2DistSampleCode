python ../train.py  --model_name=resnet50 --dataset=cifar10 --batch_size=32 --task_name=ps --task_index=0 --ps_hosts=127.0.0.1:10000 --worker_hosts=127.0.0.1:11000 --chief_host=127.0.0.1:12000 &

python ../train.py  --model_name=resnet50 --dataset=cifar10 --batch_size=32 --task_name=worker --task_index=0 --ps_hosts=127.0.0.1:10000 --worker_hosts=127.0.0.1:11000 --chief_host=127.0.0.1:12000 &

python ../train.py  --model_name=resnet50 --dataset=cifar10 --batch_size=32 --task_name=chief --task_index=0 --ps_hosts=127.0.0.1:10000 --worker_hosts=127.0.0.1:11000 --chief_host=127.0.0.1:12000 
