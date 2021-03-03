#!/bin/bash

###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2017, 2021 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################

#set -o errexit
#set -o nounset
set -o pipefail

function get_realpath() {
    if readlink -f "$@" > /dev/null 2>&1; then
        echo $(readlink -f "$@")
    elif readlink "$@" > /dev/null 2>&1; then
        echo $(readlink "$@")
    else
        echo "$@"
    fi
}

SCRIPT=$(get_realpath "$0")
SCRIPT_NAME=$(basename "$SCRIPT")
WORK_HOME=$(dirname "$SCRIPT")
WORK_HOME=$(get_realpath "$WORK_HOME")
HOSTNAME_F=$(hostname -f)
ARCH=${ARCH:-`uname -m`}

LOG_FILE=/tmp/${SCRIPT_NAME%.*}-$(id -nu).$(date +"%Y-%m-%d-%H-%M-%S").log

# Detect OS release setting
# rhel or ubuntu
eval "OS_ID=`grep "^ID=" /etc/os-release | cut -d= -f2-`"

# compatible with centos OS
[ "$OS_ID" = "centos" ] && OS_ID="rhel"
# 7 for rhel, 16.04 for ubuntu
eval "OS_VER=`grep "^VERSION_ID=" /etc/os-release | cut -d= -f2-`"

if [ $ARCH = "ppc64le" ]; then ARCH2=ppc64el; else ARCH2=$ARCH; fi
if [ "$OS_ID" = "rhel" ]; then
    IS_RHEL=true; IS_UBUNTU=false;
    OS_DISTRO="${OS_ID}`echo "$OS_VER" | cut -d. -f-1 | sed -e 's/\.//g'`"
elif [ "$OS_ID" = "ubuntu" ]; then
    IS_RHEL=false; IS_UBUNTU=true;
    OS_DISTRO="${OS_ID}`echo "$OS_VER" | cut -d. -f-2 | sed -e 's/\.//g'`"
fi

DEFAULT_CONDA_HOME="/opt/anaconda3"
DEFAULT_CONDA_INSTALLER="https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-$(uname -m).sh"

WMLA123_CONDA_INSTALLER="https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-$(uname -m).sh"
WMLA123_OPENCE_VERSION="1.1.3"
WMLA123_CUDA_VERSION="11.0"
WMLA123_PYTHON_VERSION="3.7.9"
WMLA123_DLINSIGHTS_DEPENDENCY_LIST="OpenSSL.SSL,flask,flask_httpauth,flask_script,flask_cors,elasticsearch,requests,mongoengine,alembic,sqlalchemy,pathlib,ssl,scipy,numpy"
WMLA123_DLI_DEPENDENCY_LIST="requests,six,numpy,scipy,cv2,pathlib,lmdb,chardet,IPython,pandas,configparser,nose,lmdb,redis,gflags,zmq,asyncio,ipaddress,defusedxml,matplotlib,yaml,skimage,PIL,pickle,h5py,io,absl,dill,sklearn"

WMLA_CE_INTERNAL_URL="http://congsa.ibm.com/projects/s/spectrum_wmla/wmla-dl"

#OPENCE_INTERNAL_CONDA_CHANNEL="http://congsa.ibm.com/projects/s/spectrum_wmla/open-ce/1.0.1/"
DEFAULT_OPENCE_CONDA_CHANNEL="/opt/open-ce"

# https://github.ibm.com/open-ce/tracker/issues/3
# https://github.com/open-ce/open-ce/releases

function print_usage() {
echo "
Usage: ./$SCRIPT_NAME <command> [options...]

Basic Commands:
  install_conda             Install conda
  uninstall_conda           Uninstall conda
  clean_conda               Remove unused packages and caches
  setup_wmla123             Setup conda environments for wmla 1.2.3

Test Commands:
  test_tensorflow_cuda
  test_tensorflow_cpu
  test_pytorch_cuda
  test_pytorch_cpu

Options:
  -h, --help
  -d, --conda-home          <PATH>      Conda home directory. Default: $DEFAULT_CONDA_HOME
  -i, --conda-installer     <URL>       Conda installer url. Default: $DEFAULT_CONDA_INSTALLER
  -c, --conda-channel       <CHANNEL>   Open-CE conda channel.

Examples:
  ./$SCRIPT_NAME install_conda
  ./$SCRIPT_NAME install_conda --conda-home /opt/anaconda3 --conda-installer https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-`uname -m`.sh
  
  ./$SCRIPT_NAME setup_wmla123
  ./$SCRIPT_NAME setup_wmla123 --conda-home /opt/anaconda3 --conda-channel /opt/open-ce
"
}

function deprecated_usage() {
echo -e \
"
Basic Commands:
  setup_wmlce


Options:
  -u, --cuda-version        <Version>   Specify cuda version for dlipy3. E.g: 10.2.89, 11.0.221
  -p, --python-version      <Version>   Specify python version for dlipy3 and dlipy3-cpu. E.g: 3.7.9, 3.8.5
"
}

# For avaliable cuda and python versions:
# https://anaconda.org/anaconda/cudatoolkit
# https://anaconda.org/anaconda/python

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
LIGHT_GRAY='\033[0;37m'
DARK_GRAY='\033[1;30m'
LIGHT_RED='\033[1;31m'
LIGHT_GREEN='\033[1;32m'
YELLOW='\033[1;33m'
LIGHT_BLUE='\033[1;34m'
LIGHT_PURPLE='\033[1;35m'
LIGHT_CYAN='\033[1;36m'
WHITE='\033[1;37m'
NOCOLOR='\033[0m'

function log_info() {
    echo -e "${GREEN}[INFO] `date +%F\ %H:%M:%S` $1 ${NOCOLOR}"
}

function log_warn() {
    echo -e "${PURPLE}[WARN] `date +%F\ %H:%M:%S` $1 ${NOCOLOR}"
}

function log_error() {
    echo -e "${RED}[ERROR] `date +%F\ %H:%M:%S` $1 ${NOCOLOR}"
}

function log_progress() {
    local default_width=$(tput cols 2>/dev/null)
    [ -z "$default_width" ] && default_width=80

    local width=${COLUMNS:-$default_width}
    [ $width -gt 80 ] && width=80
    
    local separator=$(printf '%*s' "${width}" '' | tr ' ' =)
    
    echo ""
    echo -e "${YELLOW}${separator}${NOCOLOR}"
    echo -e "${YELLOW}$1 ${NOCOLOR}"
    echo -e "${YELLOW}${separator}${NOCOLOR}"
}

function path_add() {
    if [ "$PATH" = "$1" ]; then return 0; fi
    if echo "$PATH" | /bin/grep -q ":$1:"; then return 0; fi
    if echo "$PATH" | /bin/grep -q "^$1:"; then return 0; fi
    if echo "$PATH" | /bin/grep -q ":$1$"; then return 0; fi
    export PATH=$PATH:$1
}

function path_remove() {
    if [[ $PATH == "$1" ]]; then
        # Handle the case where instance is the only thing in the path
        PATH=""
    else
        PATH=${PATH//":$1:"/":"} # delete any instances in the middle
        PATH=${PATH/#"$1:"/} # delete any instance at the beginning
        PATH=${PATH/%":$1"/} # delete any instance in the at the end
    fi
}

function check_cmd() {
    local cmd=""
    for cmd in "$@"
    do
        if ! [ -x "$(command -v $cmd)" ]; then
            log_error "$cmd is not installed. Abort!"
            exit 1
        fi
    done
}

function get_elapsed_minutes() {
    local t_start=$1
    local t_end=$2
    local time_elapsed=$(( $t_end - $t_start ))
    time_elapsed=$(($time_elapsed / 60))
    echo $time_elapsed
}

function download_file() {
    local src="$1"
    local dest=$2

    rm -f $dest
    if [ -f "$src" ]; then
        cp -f $src $dest
    elif [ -x "$(command -v curl)" ]; then
        echo "+ curl -k $src -o $dest"
        curl -k $src -o $dest
    elif [ -x "$(command -v wget)" ]; then
        wget --no-check-certificate $src -O $dest
    else
        log_error "curl or wget need to be installed."
        exit 1
    fi

    local download_rc=$?
    if [ "$download_rc" != "0" ]; then
        rm -f $dest
        log_error "Failed to download $src. Abort!"
        exit 1
    fi
}

function check_file() {
    local file=$1

    if [ -f "$file" ]; then
        return 0
    fi

    if wget --no-check-certificate --spider "$file" > /dev/null 2>&1; then
        return 0
    fi

    return 1
}

function contain_element() {
    local target="$1"
    shift

    local element_list=("$@")
    local element=""
    for element in "${element_list[@]}";
    do
        if [ "$target" = "$element" ]; then
            return 0
        fi
    done

    return 1
}

function check_conda_channel() {
    local channel=$1
    
    local machine_type=$(uname -m)
    
    if [ "$machine_type" != "x86_64" ] && [ "$machine_type" != "ppc64le" ]; then
        log_error "Unsupported machine architecture. Abort!"
        exit 1
    fi
    
    if [ $machine_type = "x86_64" ]; then
        local channel_json=${channel%/}/linux-64/repodata.json
    else
        local channel_json=${channel%/}/linux-ppc64le/repodata.json
    fi

    if [ -f "$channel_json" ]; then
        return 0
    fi

    if wget --no-check-certificate --spider "$channel_json" > /dev/null 2>&1; then
        return 0
    fi

    return 1
}

function install_conda() {
    log_progress "Install conda to ${OPT_CONDA_HOME}"

    if [ "$OPT_FORCE" != "true" ]; then
        if $CONDA_CMD --version > /dev/null 2>&1; then
            log_info "conda is already installed. skipping"
            return
        fi
    fi

    rm -rf ${OPT_CONDA_HOME}
    if [ -d "$OPT_CONDA_HOME" ]; then
        log_error "Failed to delete directory $OPT_CONDA_HOME. Exiting."
        exit 1
    fi
    
    rm -f /tmp/anaconda.sh
    
    OPT_CONDA_INSTALLER=${OPT_CONDA_INSTALLER:-$DEFAULT_CONDA_INSTALLER}
    
    if [[ "$OPT_CONDA_INSTALLER" == *\.tgz ]]; then
        # conductor bundle anaconda as tgz file
        # $EGO_TOP/conductorspark/conf/anaconda/distributions/Anaconda4.8.3-Python3-Linux-ppc64le/Anaconda4.8.3-Python3-Linux-ppc64le.tgzle.tgz
        # $EGO_TOP/conductorspark/conf/anaconda/distributions/Anaconda4.8.3-Python3-Linux-x86_64/Anaconda4.8.3-Python3-Linux-x86_64.tgz
        local tar_folder="/tmp/anaconda"
        rm -rf $tar_folder
        mkdir -p $tar_folder
        download_file $OPT_CONDA_INSTALLER $tar_folder/anaconda.tgz
        tar xzvf $tar_folder/anaconda.tgz -C $tar_folder
        mv $tar_folder/Anaconda*.sh /tmp/anaconda.sh
        rm -rf $tar_folder
    else
        download_file $OPT_CONDA_INSTALLER /tmp/anaconda.sh
    fi
    
    chmod +x /tmp/anaconda.sh
    /bin/bash /tmp/anaconda.sh -f -b -p ${OPT_CONDA_HOME}
    local install_rc=$?
    rm -f /tmp/anaconda.sh
    
    if [ "$install_rc" = "0" ]; then
        #$CONDA_CMD config --system --add channels defaults
        $CONDA_CMD config --system --set auto_update_conda false
        #echo "conda==4.9.2" > $OPT_CONDA_HOME/conda-meta/pinned
        #$CONDA_CMD update --all --yes
        #$CONDA_CMD clean -tipsy
        #$CONDA_CMD config --set channel_priority strict
        $CONDA_CMD --version
    else
        log_error "Failed to install conda. Abort!"
        exit 1
    fi
}

function uninstall_conda() {
    log_progress "Uninstall conda ${OPT_CONDA_HOME}"
    path_remove ${OPT_CONDA_HOME}/bin
    
    rm -rf ${OPT_CONDA_HOME}
    if [ -d "$OPT_CONDA_HOME" ]; then
        log_error "Failed to delete directory $OPT_CONDA_HOME. Exiting."
        exit 1
    fi
}

function clean_conda() {
    $CONDA_CMD clean --all --yes
}

function remove_conda_env(){
    local conda_env_name="$1"

    if [ -z "$conda_env_name" ]; then
        log_error "Empty conda env name. Abort!"
        exit 1;
    fi

    if $CONDA_CMD env list | grep -q "^$conda_env_name "; then
        log_info "Remove conda environment $conda_env_name"
        $CONDA_CMD remove --name $conda_env_name --all --yes
        
        #rm -rf /$OPT_CONDA_HOME/envs/$conda_env_name
        if $CONDA_CMD env list | grep "^$conda_env_name "; then
            log_error "Failed to remove conda env $conda_env_name. Abort!"
            exit 1;
        fi
    fi
    
    rm -rf $OPT_CONDA_HOME/envs/$conda_env_name

    if [ -d "$OPT_CONDA_HOME/envs/$conda_env_name" ]; then
        log_error "Failed to delete directory $OPT_CONDA_HOME/envs/$conda_env_name. Abort!"
        exit 1;
    fi
}

function parse_conda_env_name() {
    local conda_env_yaml=$1
    local conda_env_name=$(awk '/^name:/ {print $2}' $conda_env_yaml 2>/dev/null)
    echo $conda_env_name
}

function create_conda_env() {
    local conda_env_name="$1"
    local conda_python_version="$2"

    log_progress "Create conda env name=$conda_env_name, pythonversion=$conda_python_version"

    if [ -z "$conda_env_name" ]; then
        log_error "Empty conda env name. Abort!"
        exit 1;
    fi
    
    remove_conda_env $conda_env_name $conda_python_version

    $CONDA_CMD create --name $conda_env_name --yes pip python=$conda_python_version
}

function create_conda_env_from_yaml() {
    local conda_env_yaml="$1"
    
    if [ ! -f "$conda_env_yaml" ]; then
        log_error "Unable to read file $conda_env_yaml. Abort!"
        exit 1
    fi

    local conda_env_name=$(parse_conda_env_name $conda_env_yaml)
    if [ -z "$conda_env_name" ]; then
        log_error "Failed to parse conda environment name from $conda_env_yaml. Abort!"
        exit 1
    fi

    log_progress "Create conda env <$conda_env_name>"

    log_info "cat $conda_env_yaml"
    cat $conda_env_yaml
    echo

    remove_conda_env $conda_env_name

    local create_cmd="$CONDA_CMD env create --force -f $conda_env_yaml"
    log_info "Running $create_cmd"
    $create_cmd
    local create_rc=$?
    if [ "$create_rc" != "0" ]; then
        log_error "Failed to create conda env $conda_env_name. Error code $create_rc"
        exit 1
    fi

    log_info "List conda env <$conda_env_name> packages"
    $CONDA_CMD list -n $conda_env_name

    log_info "Check python version in conda env <$conda_env_name>"
    $OPT_CONDA_HOME/envs/$conda_env_name/bin/python --version
    if [ -f $OPT_CONDA_HOME/envs/$conda_env_name/bin/pip ]; then
        $OPT_CONDA_HOME/envs/$conda_env_name/bin/pip --version
    fi
}

function check_dependencies() {
    local conda_env_name="$1"
    local dependency_list="$2"

    if [ -n "$dependency_list" ]; then
        log_info "Check dependencies in env <$conda_env_name>"
        echo "+ $OPT_CONDA_HOME/envs/$conda_env_name/bin/python -c \"import $dependency_list\""
        $OPT_CONDA_HOME/envs/$conda_env_name/bin/python -c "import $dependency_list"
        local check_rc=$?
        if [ "$check_rc" = "0" ]; then
            echo -e "${PURPLE}Pass :)${NOCOLOR}"
        else
            log_error "Failed to check dependencies in env <$conda_env_name>"
            exit 1
        fi
    fi
}

function check_installed_opence() {
    local conda_env_name=$1
    local opence_version="$2"

    log_info "Check installed open-ce packages"

    if [ -z "$conda_env_name" ]; then
        log_error "Empty conda env name. Abort!"
        exit 1;
    fi

    local temp_list=`mktemp`
    $CONDA_CMD list -n $conda_env_name --show-channel-urls 2>/dev/null > $temp_list

    local opence_11x_list=("1.1.0" "1.1.1" "1.1.2" "1.1.3")
    if contain_element "$opence_version" "${opence_11x_list[@]}"; then
        local opence_pkg_list=(\
            "tensorflow-cpu" "pytorch-cpu" "torchvision-cpu" "py-xgboost-cpu" \
            "tensorflow" "tensorflow-datasets" "tensorboard" "pytorch" "torchvision" "opencv" "py-xgboost" \
            "tensorflow-serving" "tensorflow-probability" "tensorflow-estimator" "tensorflow-hub" "tensorflow-text" "tensorflow-model-optimization" "tensorflow-addons" "tensorflow-metadata" \
            "libevent"  "dm-tree" \
            )
    else
        local opence_pkg_list=()
    fi

    local not_installed_list=()
    for package_name in ${opence_pkg_list[@]};
    do
        local package_line=$(grep "^${package_name} " ${temp_list})
        if [ -n "$package_line" ]; then
            #echo "$package_name is installed"
            if [[ "$package_line" = *$OPT_CONDA_CHANNEL ]]; then
                echo -e "$(printf "%-40s " $package_name) ${GREEN}[OK]${NOCOLOR}"
            else
                if [ "$IS_USER_CHANNEL" = "true" ]; then
                    echo -e "$(printf "%-40s " $package_name) ${PURPLE}[WrongChannel]${NOCOLOR}"
                else
                    log_error "$package_name is not installed from $OPT_CONDA_CHANNEL"
                    exit 1
                fi
            fi
        else
            not_installed_list+=("$package_name")
        fi
    done

    #for package_name in ${not_installed_list[@]};
    #do
    #    echo -e "$(printf "%-40s " $package_name) [NotInstalled]"
    #done

    rm -rf $temp_list
}

function setup_opence_conda_channel() {
    local internal_archive="$1"
    local opence_version="$2"

    if [ -n "$OPT_CONDA_CHANNEL" ]; then
        if ! check_conda_channel $OPT_CONDA_CHANNEL; then
            log_error "Invalid user-input channel: The channel is not accessible or is invalid. channel url: $OPT_CONDA_CHANNEL"
            exit 1
        fi
    else
        if wget --no-check-certificate --spider "$WMLA_CE_INTERNAL_URL" > /dev/null 2>&1; then
            # setup conda channel if user is in ibm internal network
            if check_file "$internal_archive"; then
                OPT_CONDA_CHANNEL=$DEFAULT_OPENCE_CONDA_CHANNEL

                # Running from ibm network, setup conda channel with wmla internal archive
                log_progress "Setup open-ce conda channel"
                
                local wmla_archive="/tmp/wmla-dl.tgz"
            
                log_info "Donwload internal open-ce conda archive"
                download_file $internal_archive $wmla_archive
            
                log_info "Extract internal open-ce conda archive"
                rm -rf $OPT_CONDA_CHANNEL
                mkdir -p $OPT_CONDA_CHANNEL
                tar xvf $wmla_archive -C $OPT_CONDA_CHANNEL
                rm -rf $wmla_archive

                if ! check_conda_channel $OPT_CONDA_CHANNEL; then
                    log_error "Invalid IBM internal channel: The channel is not accessible or is invalid. channel url: $OPT_CONDA_CHANNEL"
                    exit 1
                fi

            else
                log_error "Internal conda archive $internal_archive does not exist. Abort"
                exit 1
            fi
        else
            # unreachable to ibm internal network
            log_error "Option --conda-channel <existing-channel> is required"
            exit 1
        fi
    fi

    if [ -f "$OPT_CONDA_CHANNEL/version.txt" ]; then
        log_info "Check open-ce version"
        cat "$OPT_CONDA_CHANNEL/version.txt"
    fi
}

function get_internal_conda_archive() {
    local opence_version="$1"
    local cuda_version="$2"
    local python_version="$3"

    local cuda_part1=$(echo $cuda_version | awk -F'.' '{print $1}')
    local cuda_part2=$(echo $cuda_version | awk -F'.' '{print $2}')

    local py_part1=$(echo $python_version | awk -F'.' '{print $1}')
    local py_part2=$(echo $python_version | awk -F'.' '{print $2}')

    local conda_archive="${WMLA_CE_INTERNAL_URL}/${opence_version}/wmladl-cuda${cuda_part1}.${cuda_part2}-py${py_part1}${py_part2}_${ARCH}.tgz"
    echo "$conda_archive"
}

function setup_wmla123() {
    local t_start=$(date +%s)

    # for open-ce conda channel
    local opence_version=$WMLA123_OPENCE_VERSION
    local cuda_version=${OPT_CUDA_VERSION:-$WMLA123_CUDA_VERSION}
    local python_version=${OPT_PYTHON_VERSION:-$WMLA123_PYTHON_VERSION}
    local conda_archive=$(get_internal_conda_archive $opence_version $cuda_version $python_version)
    setup_opence_conda_channel $conda_archive $opence_version

    # install conda
    # wmla 1.2.3 / conductor 2.5.0 bundles anaconda-2019.03 and miniconda-4.8.3 https://bldsrv01.eng.platformlab.ibm.com/bldapp/status.jsp?task_id=1452
    OPT_CONDA_INSTALLER=${OPT_CONDA_INSTALLER:-$WMLA123_CONDA_INSTALLER}
    install_conda

    # create conda envrionments from yaml file
    for yaml_folder in "$WORK_HOME/wmla_123"
    do
        if [ ! -d "$yaml_folder" ]; then
            log_error "Unable to read folder $yaml_folder. Abort!"
            exit 1
        fi
        
        find $yaml_folder -name '*.yaml' -o -name '*.yml' | while read conda_env_yaml; do
            local conda_env_name=$(parse_conda_env_name $conda_env_yaml)

            # Need fix from https://github.com/conda/conda/issues/8675 to specify channel priority
            # Need fix from https://github.com/conda/conda/issues/10407 to support channel for each package
            local yaml_file="/tmp/$(id -nu)-$(basename $conda_env_yaml)"
            sed -e "s#@opence-channel@#$OPT_CONDA_CHANNEL#g" $conda_env_yaml > $yaml_file

            if [ "$conda_env_name" == "dlipy3" ] || [ "$conda_env_name" == "dlipy3-cpu" ]; then
                if [ -n "$OPT_CUDA_VERSION" ]; then
                    sed -i -e "s#^- defaults::cudatoolkit=.*#- defaults::cudatoolkit==$OPT_CUDA_VERSION#g" $yaml_file
                fi

                if [ -n "$OPT_PYTHON_VERSION" ]; then
                    sed -i -e "s#^- defaults::python=.*#- defaults::python==$OPT_PYTHON_VERSION#g" $yaml_file
                fi

                create_conda_env_from_yaml $yaml_file
                check_installed_opence $conda_env_name $opence_version
                check_dependencies $conda_env_name $WMLA123_DLI_DEPENDENCY_LIST

            elif [ "$conda_env_name" = "dlinsights" ]; then
                create_conda_env_from_yaml $yaml_file
                check_dependencies $conda_env_name $WMLA123_DLINSIGHTS_DEPENDENCY_LIST

                # for CVE
                #local cve_file=$(find $OPT_CONDA_HOME/envs/dlinsights/lib/python*/site-packages/werkzeug/debug/shared/jquery.js 2>/dev/null)
                #[ -f "$cve_file" ] && mv ${cve_file} ${cve_file}.bak
                #local cve_file=$(find $OPT_CONDA_HOME/envs/dlinsights/lib/python*/site-packages/werkzeug/debug/shared/debugger.js 2>/dev/null)
                #[ -f "$cve_file" ] && mv ${cve_file} ${cve_file}.bak
            else
                create_conda_env_from_yaml $yaml_file
            fi
        done
    done

    # https://github.ibm.com/platformcomputing/wmla-tracker/issues/490
    #for conda_env_name in dlinsights dlipy3 dlipy3-cpu
    #do
    #    local jquery_file=$(find $OPT_CONDA_HOME/envs/$conda_env_name/lib/python*/site-packages/werkzeug/debug/shared/ -name jquery.js 2>/dev/null | head -n1)
    #    if [ -f "$jquery_file" ]; then
    #        mv $jquery_file $jquery_file.bak
    #    fi
    #done

    # summary
    local t_end=$(date +%s)
    local elapsed_minutes=$(get_elapsed_minutes $t_start $t_end)
    log_progress "Finished. Elapsed time: $elapsed_minutes minutes. Check $LOG_FILE for details."
    echo
}

function setup_wmlce() {
    local t_start=$(date +%s)

    # for wml-ce conda channel
    export IBM_POWERAI_LICENSE_ACCEPT=yes

    # install conda
    install_conda

    # create conda envrionments from yaml file
    for yaml_folder in "$WORK_HOME/wmlce"
    do
        if [ ! -d "$yaml_folder" ]; then
            log_error "Unable to read folder $yaml_folder. Abort!"
            exit 1
        fi
        
        find $yaml_folder -name '*.yaml' -o -name '*.yml' | while read conda_env_yaml; do
            local yaml_file="/tmp/$(id -nu)-$(basename $conda_env_yaml)"
            cp -f $conda_env_yaml $yaml_file
            create_conda_env_from_yaml $yaml_file
        done
    done

    # summary
    local t_end=$(date +%s)
    local elapsed_minutes=$(get_elapsed_minutes $t_start $t_end)
    log_progress "Finished. Elapsed time: $elapsed_minutes minutes. Check $LOG_FILE for details."
    echo
}

function init_test() {
    local conda_env_name=$1

    if ! source $OPT_CONDA_HOME/bin/activate $conda_env_name; then
        log_error "Failed to activate conda env $conda_env_name. Abort!"
        exit 1
    fi

    local test_data_folder="/tmp/wmla-conda-$(id -nu)"
    rm -rf $test_data_folder

    export DATA_DIR="$test_data_folder/data"
    export RESULT_DIR="$test_data_folder/result"

    mkdir -p $test_data_folder $DATA_DIR $RESULT_DIR $RESULT_DIR/model
}

function test_tensorflow_cuda() {
    local conda_env_name="$1"
    conda_env_name=${conda_env_name:-"dlipy3"}

    # To use single gpu:    export CUDA_VISIBLE_DEVICES=0
    # To use multiple gpu:  unset CUDA_VISIBLE_DEVICES
    init_test "$conda_env_name"

    log_progress "Test tensorflow mnist cuda. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\""
    python $WORK_HOME/models/tf2x_mnist/main.py
}

function test_tensorflow_cpu() {
    local conda_env_name="$1"
    conda_env_name=${conda_env_name:-"dlipy3-cpu"}

    init_test "$conda_env_name"

    log_progress "Test tensorflow mnist cpu"
    python $WORK_HOME/models/tf2x_mnist/main.py --no-cuda
}

function test_pytorch_cuda() {
    local conda_env_name="$1"
    conda_env_name=${conda_env_name:-"dlipy3"}

    init_test "$conda_env_name"

    log_progress "Test pytorch mnist cuda. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\""
    python $WORK_HOME/models/pytorch_mnist/main.py

    #log_progress "Test pytorch resnet cuda. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    #python $WORK_HOME/models/pytorch_mnist/main.py
}

function test_pytorch_cpu() {
    local conda_env_name="$1"
    conda_env_name=${conda_env_name:-"dlipy3-cpu"}

    init_test "$conda_env_name"

    export OMP_NUM_THREADS=1

    log_progress "Test pytorch mnist cpu"
    python $WORK_HOME/models/pytorch_mnist/main.py --no-cuda

    #log_progress "Test pytorch resnet cpu"
    #python $WORK_HOME/models/pytorch_resnet/main.py --no-cuda
}

function check_input_arg() {
    local key="$1"
    local value="$2"
    
    if [ -z "$value" ]; then
        log_error "Input is required for option $key. Abort!"
        exit 1
    fi

    if [[ $value == -* ]]; then
        log_error "Input is required for option $key. Abort!"
        exit 1
    fi
}

function invoke_cmd() {
    # https://stackoverflow.com/questions/402377/using-getopts-to-process-long-and-short-command-line-options
    # https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
    POSITIONAL=()
    while [[ $# -gt 0 ]]
    do
        local key="$1"
        case $key in
            -d|--conda-home)
                check_input_arg "$key" "$2"
                OPT_CONDA_HOME="$2"
                shift
                shift
                ;;
            -i|--conda-installer)
                check_input_arg "$key" "$2"
                OPT_CONDA_INSTALLER="$2"
                shift
                shift
                ;;
            -c|--conda-channel)
                check_input_arg "$key" "$2"
                OPT_CONDA_CHANNEL="$2"
                shift
                shift
                ;;
            -u|--cuda-version)
                check_input_arg "$key" "$2"
                OPT_CUDA_VERSION="$2"
                shift
                shift
                ;;
            -p|--python-version)
                check_input_arg "$key" "$2"
                OPT_PYTHON_VERSION="$2"
                shift
                shift
                ;;
            -f|--force)
                OPT_FORCE="true"
                shift
                ;;
            -h|--help)
                print_usage && exit 0
                ;;
            -*|--*) # unsupported flags
                log_error "unknown option \"${key}\""
                print_usage && exit 1
                ;;
            *) # unknown option
                POSITIONAL+=("$1") # save it in an array for later
                shift
                ;;
        esac
    done
    set -- "${POSITIONAL[@]}" # restore positional parameters

    # init arguments
    OPT_CONDA_HOME=${OPT_CONDA_HOME:-$DEFAULT_CONDA_HOME}
    OPT_CONDA_HOME=${OPT_CONDA_HOME%/}
    CONDA_CMD=${OPT_CONDA_HOME}/bin/conda

    OPT_FORCE=${OPT_FORCE:-"false"}

    if [ -n "$OPT_CONDA_CHANNEL" ]; then
        IS_USER_CHANNEL="true"
        OPT_CONDA_CHANNEL=${OPT_CONDA_CHANNEL%/}
    else
        IS_USER_CHANNEL="false"
    fi

    local input_cmd="$1"
    shift
    
    if [ "x$input_cmd" = "x" ]; then
        print_usage
        exit 1
    fi
    
    if [ "$input_cmd" = "help" ]; then
        print_usage
        exit 0
    fi

    local sanitize_cmd=${input_cmd//-/_}

    # check arguments
    if ! grep -q "^function $sanitize_cmd()" $SCRIPT; then
        log_error "unknown command \"$input_cmd\""
        print_usage
        exit 1
    fi

    if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "ppc64le" ]; then
        log_error "unsupported machine architecture. Abort!"
        exit 1
    fi
    
    check_cmd "wget" "sudo" "gettext" "bzip2" "zip"

   # for dlinsights
    if $IS_UBUNTU; then
        if [ ! -f /usr/include/xlocale.h ]; then
            sudo apt-get update
            sudo apt-get -y install build-essential
            sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
        fi
    fi
    
    # calling command
    $sanitize_cmd "$@"
}

invoke_cmd "$@"
