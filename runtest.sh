#!/bin/bash
cd tests

run_tests() {
  local test_entry=$1

  if [ "$test_entry" == "all" ] || [ -z "$test_entry" ]; then
    echo "Running all tests..."
    python -m pytest -s

  else
    echo "Running test: $test_entry"
    python -m pytest -v -k "$test_entry" -s
  fi

}

run_tests "$1"