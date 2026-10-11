#!/usr/bin/env bash
# 同步上游（fork 的原始仓库）并审阅其改动
#
# 用法：
#   tools/sync_upstream.sh              # 抓取 + 打印审阅信息 + 试合并（不自动提交）
#   tools/sync_upstream.sh --no-fetch   # 不联网，直接用已有的远程跟踪引用审阅
#   tools/sync_upstream.sh --apply      # 冲突已解决后再跑，完成提交与推送
#
# 约定（见 README「参与开发」）：
#   origin   = 你的 fork      —— 推送目标
#   upstream = 原始仓库       —— 只读，仅用于同步
#
# 为什么需要脚本：fork 工作流里最容易出事的不是合并本身，而是 remote 命名
# 反了导致误推到别人的仓库。这个脚本每次都先校验 remote 指向。
set -euo pipefail

cd "$(dirname "$0")/.."

UPSTREAM_NAME="${UPSTREAM_NAME:-upstream}"
ORIGIN_NAME="${ORIGIN_NAME:-origin}"
APPLY=0
FETCH=1
for arg in "$@"; do
  case "$arg" in
    --apply)    APPLY=1 ;;
    --no-fetch) FETCH=0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- remote 校验
have_remote() { git remote get-url "$1" >/dev/null 2>&1; }

if ! have_remote "$UPSTREAM_NAME"; then
  echo "!! 缺少 remote: $UPSTREAM_NAME" >&2
  echo "   添加方式： git remote add $UPSTREAM_NAME <原始仓库地址>" >&2
  exit 1
fi
if ! have_remote "$ORIGIN_NAME"; then
  echo "!! 缺少 remote: $ORIGIN_NAME（你的 fork）" >&2
  echo "   添加方式： git remote add $ORIGIN_NAME <你的 fork 地址>" >&2
  exit 1
fi

echo "== remote 指向确认 =="
echo "  $ORIGIN_NAME   (push) -> $(git remote get-url --push "$ORIGIN_NAME")"
echo "  $UPSTREAM_NAME (fetch) -> $(git remote get-url "$UPSTREAM_NAME")"

# origin 的 push 地址必须不是 upstream，否则等于往别人的仓库推
origin_push="$(git remote get-url --push "$ORIGIN_NAME")"
upstream_url="$(git remote get-url "$UPSTREAM_NAME")"
# 去掉镜像代理前缀后比较仓库路径
strip_proxy() { echo "$1" | sed -E 's#^https?://[^/]+/(https?://)?##'; }
if [[ "$(strip_proxy "$origin_push")" == "$(strip_proxy "$upstream_url")" ]]; then
  echo "!! $ORIGIN_NAME 的推送地址与 $UPSTREAM_NAME 相同 —— 会误推到别人的仓库，已中止" >&2
  exit 1
fi

# ---------------------------------------------------------------- 抓取
if [[ "$FETCH" == "1" ]]; then
  echo
  echo "== 抓取上游 =="
  if ! git fetch "$UPSTREAM_NAME" --prune; then
    echo "!! 抓取 $UPSTREAM_NAME 失败（网络/代理/地址问题）。" >&2
    echo "   当前地址：$(git remote get-url "$UPSTREAM_NAME")" >&2
    exit 1
  fi
  if ! git fetch "$ORIGIN_NAME" --prune; then
    echo "!! 抓取 $ORIGIN_NAME 失败（网络/代理/地址问题）。" >&2
    echo "   当前地址：$(git remote get-url "$ORIGIN_NAME")" >&2
    echo "   可以跳过：加 --no-fetch，仅用已有的远程跟踪引用审阅" >&2
    exit 1
  fi
else
  echo
  echo "== 跳过抓取（--no-fetch），使用已有的 $UPSTREAM_NAME/main =="
fi

UP_SHA="$(git rev-parse "$UPSTREAM_NAME/main" 2>/dev/null || echo '')"
if [[ -z "$UP_SHA" ]]; then
  echo "!! 上游没有 main 分支" >&2
  exit 1
fi

# ---------------------------------------------------------------- 审阅
BASE="$(git merge-base HEAD "$UPSTREAM_NAME/main" 2>/dev/null || echo '')"
echo
echo "== 分叉点：$(git log --oneline -1 "${BASE:-HEAD}" 2>/dev/null || echo 无共同祖先) =="

BEHIND="$(git rev-list --count HEAD.."$UPSTREAM_NAME/main" 2>/dev/null || echo 0)"
AHEAD="$(git rev-list --count "$UPSTREAM_NAME/main"..HEAD 2>/dev/null || echo 0)"
echo "   上游领先 $BEHIND 个提交 / 本仓库领先 $AHEAD 个提交"

if [[ "$BEHIND" == "0" ]]; then
  echo "   已是最新，无需合并。"
  exit 0
fi

echo
echo "== 上游新增提交 =="
git log --oneline --no-decorate HEAD.."$UPSTREAM_NAME/main"

echo
echo "== 每个提交改动的文件 =="
for c in $(git rev-list --reverse HEAD.."$UPSTREAM_NAME/main"); do
  echo "--- $(git log --oneline -1 --no-decorate "$c")"
  git show --stat --format="" "$c" | sed 's/^/    /'
done

echo
echo "== 合并后相对上游的净差异（你的改动）==="
git diff --stat "$UPSTREAM_NAME/main"...HEAD | tail -n 1

# ---------------------------------------------------------------- 冲突预演
echo
echo "== 合并冲突预演（不改动工作区）=="
CONFLICTS="$(git merge-tree --write-tree --name-only HEAD "$UPSTREAM_NAME/main" 2>&1 \
  | sed -n '/^[0-9a-f]\{40\}$/,$p' | tail -n +2 || true)"
if [[ -z "$CONFLICTS" ]]; then
  echo "   可自动合并，无冲突"
else
  echo "   以下文件需要人工处理："
  echo "$CONFLICTS" | sed 's/^/     /'
  echo
  echo "   逐个查看冲突内容："
  echo "$CONFLICTS" | sed 's/^/     git checkout --ours -- <file>   # 或 --theirs，或手工编辑/'
fi

# ---------------------------------------------------------------- 实际合并
echo
if [[ "$APPLY" != "1" ]]; then
  echo "== 预演结束。确认后执行：git merge $UPSTREAM_NAME/main =="
  echo "   冲突解决并验证通过后，用 tools/sync_upstream.sh --apply 完成提交与推送"
  exit 0
fi

echo "== 执行合并 =="
git merge "$UPSTREAM_NAME/main" --no-edit || {
  echo
  echo "!! 合并产生冲突，已停在合并中。解决后："
  echo "     git add <已解决的文件>"
  echo "     uv run pytest tests -q -m 'not network'"
  echo "     uv run python tools/sync_requirements.py --check"
  echo "     git commit      # 完成合并提交（先别 push）"
  echo "     tools/sync_upstream.sh --apply"
  exit 1
}

echo "== 合并成功，先验证 =="
uv run pytest tests -q -m "not network" || {
  echo "!! 测试未通过，已保留合并结果但未推送。请修复后手动 git commit / git push" >&2
  exit 1
}
uv run python tools/sync_requirements.py --check || {
  echo "!! requirements 与 pyproject 不一致，未推送" >&2
  exit 1
}

echo "== 推送 =="
git push "$ORIGIN_NAME" HEAD:main
echo "== 完成 =="
