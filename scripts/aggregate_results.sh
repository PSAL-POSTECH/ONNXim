#!/bin/bash

# DATAES=`ls workspace2/`
# for date in $DATAES; do
#     for config in `ls workspace2/$date`; do
#     MODEL=($(cut -d- -f1 <<< $config))
#     CONFIG=($(cut -d- -f2 <<< $config))
#     grep " finish at" workspace2/${date}/${config}/log.out | while read line; do
#       CYCLE=`echo $line | awk '{print $8}'`
#       LAYER=`echo $line | awk '{print $5}'`
#       echo "$date,$MODEL,$CONFIG,$LAYER,$CYCLE"
#     done
      
#       # grep " Model finish  at " workspace/${date}/${config}/log.out | awk '{print $}'
      
      
#     done
# done

DATAES=`ls workspace/`
for date in $DATAES; do
    for model in `ls workspace/$date`; do
      for config in `ls workspace/$date/$model`; do
        CONFIG=($(cut -d- -f2 <<< $config))
        grep " finish  at " workspace/${date}/$model/${config}/log.out | tail -n1 | while read line; do
          CYCLE=`echo $line | awk '{print $7}'`
          MODEL=`echo $line | awk '{print $5}'`
          LAYER=`echo $line | awk '{print $6}'`
          echo "$date,$model,$config,$CYCLE"
          # echo $line
        done
    done
      
      # grep " Model finish  at " workspace/${date}/${config}/log.out | awk '{print $}'
      
      
    done
done





