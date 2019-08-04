# .bashrc

force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
    # We have color support; assume it's compliant with Ecma-48
    # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
    # a case would tend to support setf rather than setaf.)
    color_prompt=yes
    else
    color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
export PS1=(caffe.img)$PS1
# added by Anaconda2 installer
export PATH="/home/nileshk/anaconda2/bin:$PATH"
#alias htop='/home/gsigurds/htop/bin/htop'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nileshk/local/lib/:/usr/local/cuda/lib64/
export CUDA_PATH=/usr/local/cuda/
alias ls='ls --color=auto'
alias gpus='watch -n 0.2 python ~/gputop.py  --no-color $@'
#source activate caffe
#alias start_nyu_fast_rcnn = 'export PYTHONPATH=/home/nileshk/Research3/original_code/faster_rcnn/fast-rcnn/caffe-fast-rcnn/python/'
#export PYTHONPATH=/home/nileshk/caffe/caffe/python:$PYTHONPATH
