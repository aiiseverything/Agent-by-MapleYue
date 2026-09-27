PY ?= python3

.PHONY: help test test-v demo lint clean tree

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test: ## Run the whole test suite (offline, stdlib unittest only)
	$(PY) -m unittest discover
	$(PY) -m unittest discover -s tests -t . -v

test-q: ## Run the whole test suite, quiet
	$(PY) -m unittest discover -s tests -t .

demo: ## Run the offline end-to-end code-assistant demo (no API key needed)
	$(PY) examples/07_code_assistant.py --offline

demo-live: ## Run the same demo against a real provider (needs OPENAI_API_KEY)
	$(PY) examples/07_code_assistant.py

lint: ## Compile-check every python file
	$(PY) -m compileall -q liteagent examples tests && echo "compile OK"

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; rm -rf .runs traces

tree: ## Show the project tree
	find . -type f -name '*.py' -o -name '*.md' -o -name '*.toml' | grep -v __pycache__ | sort
