# TF2DistSampleCode

提供tensorflow 2.5.0 parameter server strategy分布式运行步骤：

**[note:]** 与tf1不同的是，tf2中有chief, ps, worker三个角色，其中chief用来做总的调度器，ps和worker任务只需要创建server并挂起即可。


#### 【本机运行】（1ps 1 worker， 使用 ami 名称为 mytf 创建实例）

(1) 下载代码
```
cd /home/ubuntu && git clone https://github.com/ChrisQiqiang/TF2DistSampleCode.git
```
(2) 使用docker启动ps, worker进程
**[note:]** 因为在tensorflow-gpu的docker镜像基础上还要安装scikit-learn等包, ami mytf的resnet这个docker image可以直接用，后续如果需要可以传到docker hub上。
```
sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name ps --network host resnet /bin/bash -c "CUDA_VISIBLE_DEVICES=-1 \
                                        python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=ps \
                                        --task_index=0 \
                                        --ps_hosts=localhost:12345  \
                                        --worker_hosts=localhost:12346 \
                                        --chief_host=localhost:12347 " &

sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name worker --network host resnet /bin/bash -c " python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=worker \
                                        --task_index=0 \
                                        --ps_hosts=localhost:12345  \
                                        --worker_hosts=localhost:12346 \
                                        --chief_host=localhost:12347 " &
```
(3) 使用docker启动chief进程
```
sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name chief --network host resnet /bin/bash -c " python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=chief \
                                        --task_index=0 \
                                        --ps_hosts=localhost:12345  \
                                        --worker_hosts=localhost:12346 \
                                        --chief_host=localhost:12347 " 
```

#### 【分布式运行】（1ps 2 worker 1 chief， 使用 ami 名称为 mytf 创建实例）
假设 host 1: 172.31.92.94   host 2: 172.31.91.224， host1上放置1 ps 1 worker, host2上放置1个worker，
**[note:]** 对于每一个机器上的计算资源，采用eager mode, 即有几个GPU就采用几个gpu并行计算，可以理解为worker以host为单位。

(1) 在host1, host2分别下载代码
```
cd /home/ubuntu && git clone https://github.com/ChrisQiqiang/TF2DistSampleCode.git
```
(2) 在host1上使用docker启动ps, worker进程<br>
**[note:]** 因为在tensorflow-gpu的docker镜像基础上还要安装scikit-learn等包, ami mytf的resnet这个docker image可以直接用，后续如果需要可以传到docker hub上。
```
sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name ps --network host resnet /bin/bash -c "CUDA_VISIBLE_DEVICES=-1 \
                                        python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=ps \
                                        --task_index=0 \
                                        --ps_hosts=172.31.92.94:12345  \
                                        --worker_hosts=172.31.92.94:12346,172.31.91.224:12346 \
                                        --chief_host=172.31.92.94:12347 " &

sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name worker --network host resnet /bin/bash -c " python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=worker \
                                        --task_index=0 \
                                        --ps_hosts=172.31.92.94:12345  \
                                        --worker_hosts=172.31.92.94:12346,172.31.91.224:12346 \
                                        --chief_host=172.31.92.94:12347 " &
```
(3) 在host2上使用docker启动worker进程， task_index 为1
```
sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name worker --network host resnet /bin/bash -c " python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=worker \
                                        --task_index=1 \
                                        --ps_hosts=172.31.92.94:12345  \
                                        --worker_hosts=172.31.92.94:12346,172.31.91.224:12346 \
                                        --chief_host=172.31.92.94:12347 " &
```
(4) 在host1上使用docker启动chief进程
```
sudo nvidia-docker run -it -v /home/ubuntu/TF2DistSampleCode:/code --name chief --network host resnet /bin/bash -c " python -W ignore /code/tf2_resnet_ps.py \
                                        --task_name=chief \
                                        --task_index=0 \
                                        --ps_hosts=172.31.92.94:12345  \
                                        --worker_hosts=172.31.92.94:12346,172.31.91.224:12346 \
                                        --chief_host=172.31.92.94:12347 " 
```



