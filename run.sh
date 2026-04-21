#curl http://127.0.0.1:11555/api/pull -d '{"name":"gemma4:e2b","stream":false}'
#./.venv/bin/python main.py experiment=e1 nav=gemma4:e2b mujoco.headless=true ollama.host=http://localhost:11555
#curl http://127.0.0.1:11555/api/pull -d '{"name":"gemma4:e4b","stream":false}'
#./.venv/bin/python main.py experiment=e1 nav=gemma4:e4b mujoco.headless=true ollama.host=http://localhost:11555
#curl http://127.0.0.1:11555/api/pull -d '{"name":"gemma4:e31b","stream":false}'
#./.venv/bin/python main.py experiment=e1 nav=gemma4:e31b mujoco.headless=true ollama.host=http://localhost:11555
echo "ministral-3:14b"
./.venv/bin/python main.py experiment=e1 nav=ministral-3:14b mujoco.headless=true ollama.host=http://127.0.0.1:11556