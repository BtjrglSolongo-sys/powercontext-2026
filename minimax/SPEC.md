# Task status filtering

Implement optional status filtering in example/tasks.py, list_tasks.

- SPEC-1. With no status, return all tasks in their original order.
- SPEC-2. With status "open" or "done", return exactly matching tasks, preserving order.
- SPEC-3. Reject any other non-None status with ValueError.
- SPEC-4. Return independent dictionaries so callers cannot mutate stored tasks.

Required context sources are this specification, example/tasks.py and
example/test_tasks.py. FeatureTests define the new behavior.
RegressionTests protect existing ordering and copy semantics.

The initial implementation deliberately lacks filtering. Run the same tests before
and after the MiniMax-generated change. Do not change test expectations.
