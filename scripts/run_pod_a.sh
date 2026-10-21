#!/usr/bin/env bash
# Pod A: runs whose timing or memory gets reported, one at a time on an idle GPU.
set -uo pipefail
cd /root/nmep_model_zhu
R=/root/runs; mkdir -p $R
D="DATA.DATA_PATH /root/cifar10"

# Peak GPU memory of one process as nvidia-smi (and gpustat) report it.
run() {  # tag cfg opts...
  local tag=$1 cfg=$2; shift 2
  python main.py --cfg $cfg --output $R/$tag --opts $D TAG $tag "$@" > $R/$tag.log 2>&1 &
  local pid=$! peak=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 1
  done
  echo "{\"tag\": \"$tag\", \"nvidia_smi_peak_mb\": $peak}" >> $R/gpu_mem.jsonl
  echo "== $tag done"
}

run lenet configs/lenet_base.yaml
run alexnet_base configs/alexnet_base.yaml
run resnet18 configs/resnet18_base.yaml
python scripts/visualize_preds.py --ckpt $R/resnet18/resnet18/ckpt_best.pth --out $R/figs/q3_2_resnet18_preds.png
python scripts/throughput.py --out $R/throughput.json > $R/throughput.log 2>&1
echo ALL_DONE
