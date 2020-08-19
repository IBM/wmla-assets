# So there


# User1 setup in $HOME
# echo "setup user1 ...."
# cd ~
# [[ -d user1 ]] || mkdir user1; 
# cd user1;
# 
# [[ -d wmla-learning-path ]] || git clone https://github.com/IBM/wmla-learning-path.git
# export PATH=/gpfs/software/wmla-p10a117/wmla_anaconda/b0p036a/anaconda/bin:$PATH
# source /gpfs/software/wmla-p10a117/wmla_anaconda/b0p036a/anaconda/envs/powerai162/./etc/profile.d/conda.sh
# conda activate base
# 
# cd wmla-learning-path
# git fetch
# git reset --hard origin/master
# # Stop start jupyter
# ps -ef |  grep -i [j]upyter-notebook.* | grep `whoami` | sed -e "s/ \{1,\}/ /g" | cut -d " " -f2 | xargs -i kill {}
# nohup jupyter notebook --ip=0.0.0.0 --allow-root --port=$1 --no-browser --NotebookApp.token='aicoc' --NotebookApp.password='' &> classlog.out &
# 
# # User2 setup in $HOME/user2
# echo "now user2 ...."
# cd ~
# [[ -d user2 ]] || mkdir user2; 
# cd user2;
# [[ -d wmla-learning-path ]] || git clone https://github.com/IBM/wmla-learning-path.git
# 
# cd ~/user2/wmla-learning-path
# git fetch
# git reset --hard origin/master
# # Stop start jupyter
# nohup jupyter notebook --ip=0.0.0.0 --allow-root --port=$2 --no-browser --NotebookApp.token='aicoc' --NotebookApp.password='' &> classlog2.out &
 

 # $1 is user_dir and user_port
user_dir=$1
user_port=$1

[[ -d $user_dir ]] || mkdir $user_dir; 
cd $user_dir

[[ -d wmla-learning-path ]] || git clone https://github.com/IBM/wmla-learning-path.git
cd wmla-learning-path
git fetch
git reset --hard origin/master
# kill existing jupyter
ps -ef |  grep -i [j]upyter-notebook.* | grep port=$1 | sed -e "s/ \{1,\}/ /g" | cut -d " " -f2 | xargs -i kill {}
# Startup 
export PATH=/gpfs/software/wmla-p10a117/wmla_anaconda/b0p036a/anaconda/bin:$PATH
source /gpfs/software/wmla-p10a117/wmla_anaconda/b0p036a/anaconda/envs/powerai162/./etc/profile.d/conda.sh
conda activate base
#
nohup jupyter notebook --ip=0.0.0.0 --allow-root --port=$user_port --no-browser --NotebookApp.token='aicoc' --NotebookApp.password=''  &> classlog.out &
