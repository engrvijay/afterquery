#!/bin/bash

# Harbor automatically provides environment variables like HARBOR_TASK_TIMEOUT or START_TIME

if [ -z "$START_TIME" ]; then
    # Fallback to file-based tracking if running locally or if variables aren't global
    START_TIME=$(cat /tmp/task_start_time 2>/dev/null || date +%s)
fi

# Default task timeout ( 2 hours = 7200 seconds) if not defined in task.toml
TIMEOUT_SEC=${HARBOR_TASK_TIMEOUT:-7200} 
CURRENT_TIME=$(date +%s)

ELAPSED=$((CURRENT_TIME - START_TIME))
REMAINING=$((TIMEOUT_SEC - ELAPSED))

if [ $REMAINING -le 0 ]; then
    echo "Time remaining: 0 minutes (0 seconds) - Timeout Imminent"
else
    MINUTES=$((REMAINING / 60))
    SECONDS=$((REMAINING % 60))
    echo "Time remaining: ${MINUTES}m ${SECONDS}s"
fi
