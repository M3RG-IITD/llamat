end=180
batch=6
start=20
gap=20
label=redp-from-scratch
for ((i = $start; i <= end; i+=$(($batch*$gap)))); do
    for ((j = 0; j <$(($batch-1)); j++)); do
        if [ $end -ge $(($i+$j*$gap)) ]; then
            sh mthf.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/$label /scratch/cse/btech/cs1200448/meditron-to-hf-weights/${label}_$(($i+$j*$gap)) $(($i+$j*$gap)) & 
        fi
    done
    if [ $end -ge $(($i+$batch*$gap-$gap)) ]; then
        sh mthf.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/$label /scratch/cse/btech/cs1200448/meditron-to-hf-weights/${label}_$(($i+$batch*$gap-$gap)) $(($i+$batch*$gap-$gap))
    fi
done

# sh mthf.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/esr-normal-lr /scratch/cse/btech/cs1200448/meditron-to-hf-weights/esr-normal-lr_200 200