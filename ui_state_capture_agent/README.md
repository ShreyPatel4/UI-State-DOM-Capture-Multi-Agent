# ui_state_capture_agent

Project skeleton for a Python-based UI state capture agent.

## Layout

```
ui_state_capture_agent/
  src/
    __init__.py
    config.py
    models.py
    storage/
      __init__.py
      base.py
      postgres.py
      minio_store.py
    agent/
      __init__.py
      task_spec.py
      planner.py
      browser.py
      capture.py
    server/
      __init__.py
      api.py
      templates/
        base.html
        flows_list.html
        flow_detail.html
      static/
        styles.css
  scripts/
    run_task.py
    export_dataset.py
  tests/
    test_models.py
    test_storage.py
    test_agent_flow.py
  dataset/
    .gitkeep
  requirements.txt
  README.md
  pyproject.toml
  .env.example
```

All files are placeholders for future implementation.
