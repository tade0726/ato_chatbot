up:
	docker compose up -d

stop:
	docker compose stop

down:
	docker compose down


zen_run_simple_index:
	# zenml connect --url http://127.0.0.1:8080
	uv run src/ato_chatbot/pipelines/simple_index_pipeline.py


format:
	black src/ato_chatbot
	isort src/ato_chatbot


streamlit:
	uv run streamlit run src/ato_chatbot/chat_interface.py

freeze:
	uv pip freeze > requirements.txt