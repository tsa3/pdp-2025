#!/bin/bash

models=(
    'ConvNeXtBase'
    'ConvNeXtLarge'
    'ConvNeXtSmall'
    'ConvNeXtTiny'
    'ConvNeXtXLarge'
    'DenseNet121'
    'DenseNet169'
    'DenseNet201'
    'EfficientNetB0'
    'EfficientNetB1'
    'EfficientNetB2'
    'EfficientNetB3'
    'EfficientNetB4'
    'EfficientNetB5'
    'EfficientNetB6'
    'EfficientNetB7'
    'EfficientNetV2B0'
    'EfficientNetV2B1'
    'EfficientNetV2B2'
    'EfficientNetV2B3'
    'EfficientNetV2L'
    'EfficientNetV2M'
    'EfficientNetV2S'
    'InceptionResNetV2'
    'InceptionV3'
    'MobileNet'
    'MobileNetV2'
    'NASNetLarge'
    'NASNetMobile'
    'ResNet50'
    'ResNet50V2'
    'ResNet101'
    'ResNet101V2'
    'ResNet152'
    'ResNet152V2'
    'VGG16'
    'VGG19'
    'Xception'
)

interval=0.05
grace_power_socket_0="/sys/class/hwmon/hwmon3/device/power1_average"
cpu_power_socket_0="/sys/class/hwmon/hwmon4/device/power1_average"
sysio_power_socket_0="/sys/class/hwmon/hwmon5/device/power1_average"
grace_power_socket_1="/sys/class/hwmon/hwmon6/device/power1_average"
cpu_power_socket_1="/sys/class/hwmon/hwmon7/device/power1_average"
sysio_power_socket_1="/sys/class/hwmon/hwmon8/device/power1_average"

for model in "${models[@]}"
do
    results_dir="results/"
    output_file="${results_dir}/power_readings_"${model}".csv"
    echo "timestamp,grace_power_socket_0,cpu_power_socket_0,sysio_power_socket_0,grace_power_socket_1,cpu_power_socket_1,sysio_power_socket_1,total_power" > $output_file

    collect_power() {
      while true; do
        timestamp=$(date +%s%3N)
        grace_0=$(cat $grace_power_socket_0)
        cpu_0=$(cat $cpu_power_socket_0)
        sysio_0=$(cat $sysio_power_socket_0)
        grace_1=$(cat $grace_power_socket_1)
        cpu_1=$(cat $cpu_power_socket_1)
        sysio_1=$(cat $sysio_power_socket_1)

        total_power=$(((grace_0 + grace_1) / 2))

        echo "$timestamp,$grace_0,$cpu_0,$sysio_0,$grace_1,$cpu_1,$sysio_1,$total_power" >> $output_file
        sleep $interval
      done
    }

    collect_power &
    collector_pid=$!

    python3 dr_hcpa_v2_2024.py --tfrec_dir data/unifesp --dataset unifesp --results "results/"${model} --batch_size 720 --epochs 200 --verbose 0 --model ${model} --exec 0 --optimizer "SGD"

    kill $collector_pid
done