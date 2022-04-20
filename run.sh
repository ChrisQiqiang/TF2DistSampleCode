#！ /bin/bash
###########################输入的参数################################
#公有IP
hosts=("44.204.158.36" "3.87.21.185") 
#私有IP
hosts_private=("172.31.85.198" "172.31.87.170")
#放置策略（'chief位置','ps位置','worker位置'）
placement=('1' '0' '1')
#1为获取gitclone（刚创建实例时）
update_code=1
install_nethogs=0

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

if [ $install_nethogs -eq 1 ];then
    for host in "${hosts[@]}"
	do
		ssh -i tf-faye.pem ubuntu@${host} "sudo apt-get install -y nethogs"
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
                            --name ${task[$i]} \
                            --network host resnet /bin/bash -c \
                             \" python /code/train.py \
                                --model_name=resnet152 \
								--dataset=cifar100 \
								--batch_size=2048 \
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
        ssh -i tf-faye.pem ubuntu@${host} " mkdir -p /tmp/logs/bandwidth/ && mkdir -p /tmp/logs/cpu/ && mkdir -p /tmp/logs/gpu/"
        ssh -i tf-faye.pem ubuntu@${host} "sudo nethogs -t -d 1 | grep python > /tmp/logs/bandwidth/${task[$i]}_${j}.txt &" &
        ssh -i tf-faye.pem ubuntu@${host} "sudo nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,pcie.link.gen.current,utilization.memory,memory.used --format=csv --loop-ms=1000 -f /tmp/logs/gpu/${task[$i]}_${j}.txt &" &
        ssh -i tf-faye.pem ubuntu@${host} "sudo mpstat 1 > /tmp/logs/cpu/${task[$i]}_${j}.txt &" &

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

sleep 10m

echo "####################################################"
echo "###############  搬运运行日志到本地  #################"
echo "####################################################"


current=`date "+%Y-%m-%d-%H:%M:%S"` 
 for host in "${hosts[@]}"
 do
    scp -r -i tf-faye.pem ubuntu@${host} /tmp/logs/ ./${current}log/
 done

