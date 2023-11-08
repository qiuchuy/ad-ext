#!/bin/bash
cd test

run_tests() {
  local test_entry = "$1"
  if [["$test_entry" == "all" or -z "$1"]]; then
     python -m pytest -s
  else
    python -m pytest -v -k "$test_entry" -s
  fi
}

run_tests "$1"