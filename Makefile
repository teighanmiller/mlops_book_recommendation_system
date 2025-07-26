test:
	pytest tests/

integration_test: test
	bash integration_test/run.sh

quality_checks:
	isort .
	black .
	pylint --recursive=y src dags tests


run: quality_checks test integration_test
	echo run
