#!/bin/bash

# script to run with NVIDIA GPU profiling tools if available

if [ $# -eq 0 ]; then
    echo "Usage: ./profile.sh <app> [args...]"
    exit 1
fi

APP=$1
shift  # remove app name, rest is passed through as args

if [ ! -x "$APP" ]; then
    echo "ERROR: $APP not found or not executable. Did you build first?"
    exit 1
fi

nsys profile --stats=true -o profile_report "$APP" "$@"