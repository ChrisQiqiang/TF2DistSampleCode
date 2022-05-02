#！ /bin/bash
###########################输入的参数################################
#公有IP
hosts=("52.55.152.203" "44.204.40.127") 
#私有IP
hosts_private=("172.31.85.198" "172.31.87.170")
#放置策略（'chief位置','ps位置','worker位置'）
placement=('1' '0' '1')
#1为获取gitclone（刚创建实例时）
update_code=0
install_tools=0

args=$1

func_train()
{
    ##########打印配置信息#######
    declare -a info
    echo "####################################################"
    echo "################  打印配置信息  #####################"
    echo "####################################################"
    role_info=("chief" "ps" "worker")

    i=0
    for role in "${placement[@]}"
    do
        indexs=($role)
        j=0
        for index in "${indexs[@]}"
        do
            info[${index}]="${info[${index}]} ${role_info[${i}]}"
            let j+=1
        done
    let i+=1
    done

    for (( i=0; i<${#hosts[@]}; i++))
    do
    info[$i]="Host ${i} INFO: pubic IP address: ${hosts[i]}, private  IP address: ${hosts_private[i]} ... Role: ${info[${i}]}"
    echo ${info[$i]}
    done

    echo ""
    #########配置信息配置结束########

    #根据放置策略生成集群配置信息
    role_hosts=()
    i=0
    for role in "${placement[@]}"
    do
        let port=5555+i
        indexs=($role)
        tmp=""
        for index in "${indexs[@]}"
        do
            # echo $index
            host=${hosts_private[$index]}:${port}
            if [ ${#tmp} -eq 0 ];then
                tmp=$host
            else
                tmp="$tmp,$host"
            fi
        done
        role_hosts[${#role_hosts[@]}]=$tmp
        let i+=1
    done

    #在每个机器上准备相应代码
    if [ $update_code -eq 1 ];then
        for host in "${hosts[@]}"
        do
            echo ..
            ssh -i tf-faye.pem ubuntu@${host} "cd /home/ubuntu && sudo rm -rf TF2DistSampleCode && git clone https://github.com/ChrisQiqiang/TF2DistSampleCode.git" 
        done
    fi

    sleep 1m

    if [ $install_tools -eq 1 ];then
        for host in "${hosts[@]}"
        do
            ssh -i tf-faye.pem ubuntu@${host} "sudo apt-get install -y nethogs sysstat"
        done
    fi

    wait

    echo "####################################################"
    echo "###############  前序工作准备完成  ##################"
    echo "####################################################"

    #启动相关task
    i=0
    task=("chief" "ps" "worker")
    for role in "${placement[@]}"
    do
        indexs=($role)
        j=0
        for index in "${indexs[@]}"
        do
            host=${hosts[$index]}
            echo ""
            echo "host: ${host} start ${task[$i]}"
            tmp=" run -v /home/ubuntu/TF2DistSampleCode:/code \
                                --gpus all \
                                --privileged=true \
                                -p 127.0.0.1:6006:6006 \
                                --name ${task[$i]} \
                                --network host tf2_image /bin/bash -c \
                                \" python /code/train.py \
                                    --model_name=resnet50 \
                                    --dataset=vgg \
                                    --batch_size=256 \
                                    --task_name=${task[$i]} \
                                    --task_index=${j} \
                                    --ps_hosts=${role_hosts[1]} \
                                    --worker_hosts=${role_hosts[2]} \
                                    --chief_host=${role_hosts[0]} \"  "
            if [ ${task[i]} == "ps" ];then
                command="sudo docker ${tmp}"
            else
                command="sudo nvidia-docker ${tmp}"
            fi         
            echo $command
            ssh -i tf-faye.pem ubuntu@${host} "sudo docker stop ${task[$i]} > /dev/null && sudo docker rm  ${task[$i]} > /dev/null" 
            if [ $i -eq 0 ];then
            ssh -i tf-faye.pem ubuntu@${host} "sleep 1m && ${command}" &
            else
            ssh -i tf-faye.pem ubuntu@${host}  "${command}" &
            fi
        let j+=1
        done
        role_hosts[${#role_hosts[@]}]=$tmp
        let i+=1
    done
}

func_monitor(){
    #启动相关task
    i=0
    for host in "${hosts[@]}"
    do
            echo "@@@ NOW MONITOR HOST ${i}: ${host} "
            #启动profiler
            ssh -i tf-faye.pem ubuntu@${host} "tensorboard --logdir=/home/ubuntu/TF2DistSampleCode/logs/ --bind_all“
            #创建新的logs文件目录
            ssh -i tf-faye.pem ubuntu@${host} "sudo rm -rf /home/ubuntu/tmp/logs/ && mkdir -p /home/ubuntu/tmp/logs/"
            #杀死已有进程
            ssh -i tf-faye.pem ubuntu@${host} "sudo kill -9 `ps -ef  | grep nethogs | awk '{print $2}'`"
            ssh -i tf-faye.pem ubuntu@${host} "sudo kill -9 `ps -ef  | grep mpstat | awk '{print $2}'`"
            ssh -i tf-faye.pem ubuntu@${host} "sudo kill -9 `ps -ef  | grep nvidia-smi | awk '{print $2}'`"
            #监控进程启动
            ssh -i tf-faye.pem ubuntu@${host} "sudo nethogs -t -d 1 | grep python > /home/ubuntu/tmp/logs/bandwidth_host${i}.txt " &
            ssh -i tf-faye.pem ubuntu@${host} "sudo nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,pcie.link.gen.current,utilization.memory,memory.used --format=csv --loop-ms=1000 -f /home/ubuntu/tmp/logs/gpu_host${i}.txt &" &
            ssh -i tf-faye.pem ubuntu@${host} "sudo mpstat 1 > /home/ubuntu/tmp/logs/cpu_host${i}.txt &" &
            let i+=1
    done

    sleep ${args}m

    echo "####################################################"
    echo "###############  搬运运行日志到本地  #################"
    echo "####################################################"


    current=`date "+%Y-%m-%d-%H-%M-%S"` 
    mkdir ./${current}log/
    for host in "${hosts[@]}"
    do
        scp -r -i tf-faye.pem ubuntu@${host}:/home/ubuntu/tmp/logs/ ./${current}log/
    done
}

if [ -z "$args" ];then
    echo "用于监控的时间参数为空，执行训练语句。" 
    func_train
else
    echo "用于监控的时间参数args:" $args
    func_monitor
fi