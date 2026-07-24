#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# 基本设置
# =========================
PROGRAM="./stlq_main"
BASE_CFG="../configs/basic_linux.cfg"
LOG_DIR="./grid_logs" 
#if need not this log, move "${cmd[@]}" >"$log_file" 2>&1 to "${cmd[@]}"

# 0: 真跑
# 1: 只打印命令，不执行
DRY_RUN=0

# 0: 某次失败后立即停止
# 1: 某次失败后继续下一个
CONTINUE_ON_ERROR=1

# =========================
# 需要做笛卡尔积的参数
# 只要写成数组即可；如果某个参数只想固定一个值，就写成单元素数组
# =========================
# legacy_root_only
# hybrid
# fast

TRAIN_INIT_LINKAGE_MODES=(
  legacy_root_only
)

BASE_ENCODE_H_BEAMS=(
  2
)

TRAIN_LINKAGE_KNN_KS=(
  15
)


BASE_LINKAGE_KNN_KS=(
  30
)


# =========================
# 这 4 个参数“按索引绑定推进”
# 它们长度必须完全一致
# 第 i 项 together 组成 1 个组合
# =========================
LINKED_TRAIN_LINKAGE_NUM_LAYERS=(
  9
)

LINKED_BASE_LINKAGE_NUM_LAYERS=(
  10
)

LINKED_TRAIN_LINKAGE_DEPTH_K=(
  6
)

LINKED_BASE_LINKAGE_DEPTH_K=(
  8
)

# =========================
# 检查
# =========================
if [[ ! -x "$PROGRAM" ]]; then
  echo "错误: 可执行文件不存在或没有执行权限: $PROGRAM"
  exit 1
fi

if [[ ! -f "$BASE_CFG" ]]; then
  echo "错误: 配置文件不存在: $BASE_CFG"
  exit 1
fi

linked_n=${#LINKED_TRAIN_LINKAGE_NUM_LAYERS[@]}

if [[ ${#LINKED_BASE_LINKAGE_NUM_LAYERS[@]} -ne $linked_n ]] || \
   [[ ${#LINKED_TRAIN_LINKAGE_DEPTH_K[@]} -ne $linked_n ]] || \
   [[ ${#LINKED_BASE_LINKAGE_DEPTH_K[@]} -ne $linked_n ]]; then
  echo "错误: 4 个绑定推进的数组长度必须一致"
  exit 1
fi

mkdir -p "$LOG_DIR"

total=$(( \
  ${#TRAIN_INIT_LINKAGE_MODES[@]} * \
  ${#BASE_ENCODE_H_BEAMS[@]} * \
  ${#BASE_LINKAGE_KNN_KS[@]} * \
  ${#TRAIN_LINKAGE_KNN_KS[@]} * \
  linked_n \
))

echo "总任务数: $total"
echo

run_id=0

# =========================
# 主循环
# =========================
for mode in "${TRAIN_INIT_LINKAGE_MODES[@]}"; do
  for h_beam in "${BASE_ENCODE_H_BEAMS[@]}"; do
    for base_knn in "${BASE_LINKAGE_KNN_KS[@]}"; do
      for train_knn in "${TRAIN_LINKAGE_KNN_KS[@]}"; do
        for ((i=0; i<linked_n; i++)); do
          run_id=$((run_id + 1))

          train_num_layers="${LINKED_TRAIN_LINKAGE_NUM_LAYERS[$i]}"
          base_num_layers="${LINKED_BASE_LINKAGE_NUM_LAYERS[$i]}"
          train_depth_k="${LINKED_TRAIN_LINKAGE_DEPTH_K[$i]}"
          base_depth_k="${LINKED_BASE_LINKAGE_DEPTH_K[$i]}"

          cmd=(
            "$PROGRAM"
            --config "$BASE_CFG"
            --set "train.init_linkage_mode=$mode"
            --set "base.encode.H_beam=$h_beam"
            --set "base.linkage.knn_k=$base_knn"
            --set "train.linkage.knn_k=$train_knn"
            --set "train.linkage.num_layers=$train_num_layers"
            --set "base.linkage.num_layers=$base_num_layers"
            --set "train.linkage.depth_k=$train_depth_k"
            --set "base.linkage.depth_k=$base_depth_k"
          )

          tag="run${run_id}_mode-${mode}_hb-${h_beam}_bck-${base_knn}_tck-${train_knn}_tnl-${train_num_layers}_bnl-${base_num_layers}_tdk-${train_depth_k}_bdk-${base_depth_k}"
          log_file="${LOG_DIR}/${tag}.log"

          echo "=================================================="
          echo "[$run_id/$total] 开始: $(date '+%Y-%m-%d %H:%M:%S')"
          echo "日志: $log_file"
          printf '命令: '
          printf '%q ' "${cmd[@]}"
          echo
          echo "=================================================="

          if [[ $DRY_RUN -eq 1 ]]; then
            echo
            continue
          fi

          set +e
          "${cmd[@]}" >"$log_file" 2>&1
          #"${cmd[@]}"
          status=$?
          set -e

          if [[ $status -ne 0 ]]; then
            echo "失败: exit code = $status"
            echo "失败日志: $log_file"
            echo

            if [[ $CONTINUE_ON_ERROR -eq 1 ]]; then
              continue
            else
              exit "$status"
            fi
          fi

          echo "完成: $(date '+%Y-%m-%d %H:%M:%S')"
          echo
        done
      done
    done
  done
done

echo "所有任务执行完成。"
