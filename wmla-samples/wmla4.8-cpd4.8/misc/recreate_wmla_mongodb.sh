#!/bin/bash

R='\033[0;31m'
G='\033[0;32m'
NC='\033[0m'
SUCCESS="${G}OK${NC}"
FAIL="${R}FAIL${NC}"

oc project
echo WARNING: This script will recreate WML-A MongoDB. Please stop all WML-A workloads and back up your data if necessary before proceeding.
while true; do
    read -p "Continue? (yes/no): " choice
    case $choice in
        [Yy]es)
	    break
            ;;
        [Nn]o)
            echo "Exiting."
            exit
            ;;
        *)
            echo "Please enter yes or no."
            ;;
    esac
done

shutdown_mongodb() {
    local timeout=300
    local start_time=$(date +%s)

    echo -e "\nShutting down MongoDB..."
    oc scale --replicas=0 sts wmla-mongodb

    echo Waiting for all MongoDB replicas to exit...
    local replicas=$(oc get sts wmla-mongodb -o=jsonpath='{.status.readyReplicas}')

    while true; do
        replicas=$(oc get sts wmla-mongodb -o=jsonpath='{.status.readyReplicas}')
        if [ "_$replicas" == "_" ]; then
            echo "Replicas is 0 for Mongodb."
            break
        fi

        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -gt $timeout ]; then
            echo -e "Shut down MongoDB - ${FAIL}"
            echo Please ensure that all Mongodb pods have exited properly before rerunning this script.
            exit 1
        fi

        sleep 5
    done
    echo -e "Shut down MongoDB - ${SUCCESS}"
}

restart_mongodb() {
    local timeout=300
    local start_time=$(date +%s)

    echo -e "\nRestarting MongoDB..."
    oc scale --replicas=3 sts wmla-mongodb

    echo Waiting for all MongoDB replicas to start...
    local replicas=$(oc get sts wmla-mongodb -o=jsonpath='{.status.readyReplicas}')

    while true; do
        replicas=$(oc get sts wmla-mongodb -o=jsonpath='{.status.readyReplicas}')
        if [ "_$replicas" == "_3" ]; then
            echo "Replicas is 3 for Mongodb."
            break
        fi

        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -gt $timeout ]; then
            echo -e "Restart MongoDB - ${FAIL}"
            echo Please ensure that all Mongodb pods have exited properly before rerunning this script.
            exit 1
        fi

        sleep 5
    done
    echo -e "Restart MongoDB - ${SUCCESS}"
}

reconnect_mongodb() {
    local timeout=300
    local start_time=$(date +%s)

    echo -e "\nRestarting WML-A dlpd, msd and watchdog pods..."
    oc get po |grep -E 'wmla-dlpd|wmla-msd|wmla-watchdog' |awk '{print $1}' |xargs oc delete po

    echo Waiting for WML-A dlpd, msd and watchdog pods to start...
    local replicas_dlpd=$(oc get deploy wmla-dlpd -o=jsonpath='{.status.readyReplicas}')
    local replicas_msd=$(oc get deploy wmla-msd -o=jsonpath='{.status.readyReplicas}')
    local replicas_watchdog=$(oc get deploy wmla-watchdog -o=jsonpath='{.status.readyReplicas}')

    while true; do
        replicas_dlpd=$(oc get deploy wmla-dlpd -o=jsonpath='{.status.readyReplicas}')
        replicas_msd=$(oc get deploy wmla-msd -o=jsonpath='{.status.readyReplicas}')
        replicas_watchdog=$(oc get deploy wmla-watchdog -o=jsonpath='{.status.readyReplicas}')
        if [ "_$replicas_dlpd" == "_1" ] && [ "_$replicas_msd" == "_1" ] && [ "_$replicas_watchdog" == "_1" ]; then
            break
        fi

        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -gt $timeout ]; then
            echo -e "Restart WML-A dlpd, msd and watchdog pods timeout - ${FAIL}"
            echo Please ensure that the pods for services dlpd, msd, and watchdog are in the correct status.
            exit 1
        fi

        sleep 5
    done
    echo -e "Restart WML-A dlpd, msd and watchdog pods - ${SUCCESS}"
}

shutdown_mongodb

echo -e "\nCleaning up mongodb data..."
oc delete pvc data-wmla-mongodb-0 data-wmla-mongodb-1 data-wmla-mongodb-2
echo -e "Clean up mongodb data - ${SUCCESS}"

restart_mongodb
reconnect_mongodb

echo All done.
