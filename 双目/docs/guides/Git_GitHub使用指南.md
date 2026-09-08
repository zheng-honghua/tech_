# Git 与 GitHub 使用指南

面向 Git 初学者的日常速查手册。示例以 Windows 的 Git Bash 为主；命令在 PowerShell、macOS 和 Linux 中大多也可使用。

## 1. Git 和 GitHub 的区别

- **Git**：安装在电脑上的版本控制工具。它记录文件的每次修改，可以在本地提交、查看历史和恢复版本。
- **GitHub**：在线代码托管平台。它保存远程仓库，方便备份、协作和分享。

简单理解：Git 是本地的“版本管理器”，GitHub 是存放 Git 仓库的“云端平台”。没有 GitHub 也能使用 Git；没有 Git 就不能正常向 GitHub 提交版本记录。

## 2. 第一次使用：配置姓名和邮箱

安装 Git 后，先设置提交记录中显示的姓名和邮箱。通常应使用 GitHub 账户绑定的邮箱。

```bash
git config --global user.name "你的名字"
git config --global user.email "your-email@example.com"
```

查看配置是否成功：

```bash
git config --global --list
```

`--global` 表示此电脑上的所有仓库默认使用这套配置。若某个项目需要使用不同身份，可在该项目目录中去掉 `--global` 后重新设置。

## 3. 创建或获取仓库

### 3.1 把已有项目变成 Git 仓库

先在 Git Bash 中进入项目文件夹，再执行：

```bash
git init
```

这会创建隐藏的 `.git` 目录，用于保存版本历史。不要随意删除 `.git`，否则项目将失去 Git 记录。

### 3.2 从 GitHub 下载已有仓库

在想存放项目的目录中执行：

```bash
git clone https://github.com/用户名/仓库名.git
```

例如：

```bash
git clone https://github.com/zheng-honghua/tech_.git
```

`clone` 会下载项目文件、完整历史，以及名为 `origin` 的远程地址配置。

## 4. 四个位置：文件如何流动

```text
工作区  -- git add -->  暂存区  -- git commit -->  本地仓库  -- git push -->  远程仓库（GitHub）
```

| 位置 | 含义 |
| --- | --- |
| 工作区 | 电脑中正在编辑的项目文件。 |
| 暂存区 | 准备在下一次提交中保存的文件清单。 |
| 本地仓库 | 当前电脑 `.git` 中保存的提交历史。 |
| 远程仓库 | GitHub 上的项目副本，通常通过 `origin` 指向。 |

`git add` 不是上传，而是把修改从工作区放进暂存区；`git commit` 也只保存到本地；`git push` 才会上传到 GitHub。

## 5. 查看、暂存与提交

### 查看当前状态

```bash
git status
```

这是最常用也最安全的检查命令。它会显示哪些文件被修改、哪些已暂存，以及当前分支是否需要推送。

### 暂存文件

暂存当前目录下所有**未被忽略**的新增和修改：

```bash
git add .
```

只暂存一个指定文件：

```bash
git add 文件名.py
```

也可以指定目录：

```bash
git add 图片目录/
```

暂存后建议再执行一次 `git status`，确认提交内容正确。

### 创建本地提交

```bash
git commit -m "说明本次修改的简短信息"
```

例如：

```bash
git commit -m "添加几何测试图片"
```

提交信息应描述“做了什么”，而不是只写“修改”“更新”。

## 6. 远程仓库：查看、添加和修改

查看已配置的远程仓库：

```bash
git remote -v
```

首次把本地仓库关联到 GitHub：

```bash
git remote add origin https://github.com/zheng-honghua/tech_.git
```

修改 `origin` 指向的新地址：

```bash
git remote set-url origin https://github.com/zheng-honghua/tech_.git
```

删除远程地址：

```bash
git remote remove origin
```

删除后如仍需要关联，再执行 `git remote add origin <地址>`。

### `remote origin already exists` 是什么？

它表示当前仓库已经有一个名为 `origin` 的远程仓库。常见于已执行过 `git remote add origin ...`，或项目是通过 `git clone` 获取的。

先检查现有地址：

```bash
git remote -v
```

若地址正确，例如已显示 `https://github.com/zheng-honghua/tech_.git`，无需再次添加，直接推送即可。

若地址不对，**正确做法通常是修改地址**：

```bash
git remote set-url origin https://github.com/zheng-honghua/tech_.git
```

只有确认需要彻底移除该远程时，才使用：

```bash
git remote remove origin
git remote add origin https://github.com/zheng-honghua/tech_.git
```

## 7. 推送到 GitHub

第一次推送当前 `main` 分支：

```bash
git push -u origin main
```

其中：

- `origin` 是远程仓库的默认名称；
- `main` 是本地分支名；
- `-u` 会建立跟踪关系。

建立跟踪关系后，之后通常只需：

```bash
git push
```

如果你的分支不是 `main`，先用下面的命令确认：

```bash
git branch --show-current
```

然后将上述命令中的 `main` 替换成实际分支名。

## 8. 获取远程更新：git pull

在开始继续编辑、或推送被拒绝时，先获取 GitHub 上其他提交：

```bash
git pull
```

它会下载并合并当前跟踪分支的远程更新。若出现冲突，Git 会标出冲突文件；处理后执行 `git add <文件>`，再 `git commit` 完成合并。

日常单人项目中，也建议在开始工作前或推送前先运行一次 `git pull`，避免遗漏网页端或其他电脑的更新。

## 9. .gitignore：不提交哪些文件

`.gitignore` 用来告诉 Git：哪些未跟踪文件不应出现在 `git add .` 中。它适合放缓存、虚拟环境、可重新生成的输出和本机采集数据，不适合放需要纳入版本管理的源代码、文档或小型固定测试夹具。

当前 Python 项目可使用：

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
*.egg-info/
output/
data/
几何测试_*/
几何混合测_*/
```

规则说明：

- 以 `/` 结尾通常表示忽略整个目录，如 `.venv/`。
- `*` 可匹配任意字符，如 `*.egg-info/`。
- `*.py[cod]` 会匹配 `.pyc`、`.pyo`、`.pyd` 等 Python 编译产物。

本仓库有意忽略`data/`、`几何测试_*/`和`几何混合测_*/`：它们包含本机拍摄数据、重复样本和生成产物，不随源码上传。需要长期加入回归测试的少量已审核图片，应复制到`tests/fixtures/`并明确暂存，而不是解除整个数据集目录的忽略规则。

注意：`.gitignore` 主要影响“未被 Git 跟踪”的文件。已提交过的文件即使后来写进 `.gitignore`，仍会继续被跟踪。

## 10. 排查被忽略文件

查看被 Git 忽略的文件：

```bash
git status --ignored
```

若某个文件确实应该提交，但暂时不能或不想修改 `.gitignore`，可以强制暂存：

```bash
git add -f 路径/文件名
```

`-f` 表示强制加入。对于需要长期提交的文件，优先修正 `.gitignore` 规则，而不是反复使用 `git add -f`。

## 11. `nothing to commit, working tree clean` 的含义

这条提示表示：

- 工作区没有未提交的修改；
- 暂存区没有等待提交的内容；
- 当前本地仓库是干净的。

这不是报错。若你刚提交过，说明提交已完成。接着可执行：

```bash
git push
```

若你本以为修改了文件却看到这条提示，请确认：是否编辑并保存了正确的项目文件夹、文件是否被 `.gitignore` 忽略、以及是否已在之前提交过。

## 12. Git Bash 中中文显示为转义字符

有时 `git status` 中的中文文件名会显示成 `\346\265\213...` 一类转义字符。这通常只是 Git 的显示方式，文件本身没有损坏。

可执行以下命令，让 Git 尽量直接显示中文路径：

```bash
git config --global core.quotepath false
```

重新打开 Git Bash 或再次运行 `git status` 查看效果。

## 13. GitHub 连接 443 被重置或无法连接

若推送时出现 `Connection reset`、`Could not connect to server`、443 连接失败等，通常是网络、代理、防火墙或 DNS 问题，并不代表 Git 仓库损坏。

先测试 GitHub 是否可访问：

```bash
curl -I https://github.com
```

查看 Git 是否设置了代理：

```bash
git config --global --get http.proxy
git config --global --get https.proxy
```

若代理地址已失效，可取消 Git 的代理配置：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

也可检查环境变量中的代理（Git Bash）：

```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

必要时切换网络、检查代理软件是否已开启且端口正确，然后重试：

```bash
git push
```

不要在网络异常时反复执行 `git init`、删除 `.git` 或重新提交；先解决连接问题即可。

## 14. 日常标准流程：修改代码或文档后上传

在项目根目录依次执行：

```bash
# 1. 查看当前改动
git status

# 2. 获取远程已有更新（首次推送前若远程为空可跳过）
git pull --rebase

# 3. 暂存本次要提交的全部内容
git add .

# 4. 再检查一次暂存内容
git status

# 5. 创建本地提交
git commit -m "描述本次修改"

# 6. 上传到 GitHub
git push
```

若只想提交部分文件，把`git add .`换成具体的`git add 文件名`。本地采集图片默认被忽略；若确实要共享完整数据集，优先使用Git LFS、GitHub Release或独立数据仓库。提交前多看一次`git status`，可避免把临时文件或不相关改动上传。

## 15. 常用撤销与恢复

先运行 `git status`，确认文件处于什么状态，再选择命令。下面按风险区分。

### 相对安全的命令

取消暂存某个文件，但保留工作区修改：

```bash
git restore --staged 文件名
```

取消暂存全部文件，但保留工作区修改：

```bash
git restore --staged .
```

查看提交历史：

```bash
git log --oneline
```

修改最近一次提交的信息（尚未推送时最适合）：

```bash
git commit --amend -m "新的提交说明"
```

### 会丢失修改，使用前务必确认

放弃某个已跟踪文件在工作区的未提交修改：

```bash
git restore 文件名
```

放弃所有已跟踪文件在工作区的未提交修改：

```bash
git restore .
```

以上两条会让文件回到最近一次提交的状态，未提交内容通常难以恢复。对不确定的改动，先复制备份或使用 `git diff` 查看。

将本地分支强制回退到某个提交：

```bash
git reset --hard 提交ID
```

这是高风险命令，会丢弃该提交之后的本地修改。已经推送并被他人使用的提交，不要随意用它改写历史。

## 16. 命令速查表

| 命令 | 作用 |
| --- | --- |
| `git status` | 查看工作区、暂存区和分支状态。 |
| `git init` | 在当前目录创建新的 Git 仓库。 |
| `git clone <地址>` | 下载远程仓库到本地。 |
| `git add .` | 暂存当前目录下所有未忽略的改动。 |
| `git add <文件>` | 暂存指定文件。 |
| `git add -f <文件>` | 强制暂存被忽略的文件。 |
| `git commit -m "说明"` | 将暂存区内容保存为本地提交。 |
| `git log --oneline` | 用简洁形式查看历史提交。 |
| `git remote -v` | 查看远程仓库地址。 |
| `git remote add origin <地址>` | 新增名为 origin 的远程仓库。 |
| `git remote set-url origin <地址>` | 修改 origin 的地址。 |
| `git remote remove origin` | 删除 origin 远程地址。 |
| `git push -u origin main` | 首次推送 main 并建立跟踪关系。 |
| `git push` | 上传本地提交到跟踪的远程分支。 |
| `git pull` | 下载并合并远程更新。 |
| `git status --ignored` | 连同被忽略文件一起查看状态。 |
| `git restore --staged <文件>` | 取消暂存，保留文件修改。 |
| `git restore <文件>` | 丢弃该文件未提交的工作区修改。 |
| `git config --global core.quotepath false` | 让 Git 优先正常显示中文文件名。 |

## 17. 最小操作口诀

每次提交前记住这六步：

```text
status → pull --rebase → add → status → commit → push
```

不确定时，先执行 `git status`；遇到远程地址问题，先执行 `git remote -v`；担心丢失修改时，不要急着运行 `restore` 或 `reset --hard`。
