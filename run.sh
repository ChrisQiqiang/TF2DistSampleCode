#！ /bin/bash
####################################################################
###########################输入的参数################################
####################################################################
hosts=("3.89.226.104" "44.201.154.221")
hosts_private=("172.31.92.174" "172.31.87.79")
placement=('0' '0' '0 1')



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
for host in "${hosts[@]}"
do
    echo ..
    ssh -i tf-faye.pem ubuntu@${host} "cd /home/ubuntu && sudo rm -rf TF2DistSampleCode && git clone https://github.com/ChrisQiqiang/TF2DistSampleCode.git"
done


启动相关task
i=0
task=("chief" "ps" "worker")
for role in "${placement[@]}"
do
    indexs=($role)
    j=0
    for index in "${indexs[@]}"
    do
        host=${hosts[$index]}
        echo "host: ${host} start ${task[$i]}"

        command="sudo nvidia-docker run -v /home/ubuntu/TF2DistSampleCode:/code \
                            --name ${task[$i]} \
                            --network host resnet /bin/bash -c \
                             \"python /code/test.py \
                                --model_name=resnet101 \
                                --task_name=${task[$i]} \
                                --task_index=${j} \
                                --ps_hosts=${role_hosts[1]}   \
                                --worker_hosts=${role_hosts[2]} \
                                --chief_host=${role_hosts[0]}\"  "
        echo $command
        ssh -i tf-faye.pem ubuntu@${host} "sudo docker stop ${task[$i]} && sudo docker rm  ${task[$i]}"
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
